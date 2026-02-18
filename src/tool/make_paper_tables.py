# src/tool/make_paper_tables.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from src.utils.paths import project_root



# -----------------------------
# Helpers
# -----------------------------
def _read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    df = pd.read_csv(p)
    if len(df) == 0:
        raise ValueError(f"Empty CSV: {p}")
    return df


def _fmt_float(x: Any, nd: int) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""


def _ensure_outdir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _reorder_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    rest = [c for c in df.columns if c not in keep]
    return df[keep + rest]


# -----------------------------
# Paper-like schemas
# -----------------------------
@dataclass(frozen=True)
class PaperFormat:
    # decimals for metrics
    metric_nd: int = 3
    auc_nd: int = 3
    # if True, keep AUC where available
    keep_auc: bool = True


PAPER = PaperFormat(metric_nd=3, auc_nd=3, keep_auc=True)

# SQI order exactly as you want to show in the paper table
SQI_ORDER = ["bSQI", "iSQI", "kSQI", "sSQI", "pSQI", "fSQI", "basSQI"]

# Table6 group order in paper
GROUP_ORDER = ["Pairs", "Triplets", "Quadruplets", "Quintuplets", "Sextuplets", "All SQI"]


# -----------------------------
# Converters
# -----------------------------
def make_table5_svm(df: pd.DataFrame, fmt: PaperFormat = PAPER) -> pd.DataFrame:
    """
    Input (your svm_tables):
      SQI, n_features, cv_acc, Ac_train, Se_train, Sp_train, Ac_test, Se_test, Sp_test
    Output (paper-like):
      SQI, n_features, Ac_train, Se_train, Sp_train, Ac_test, Se_test, Sp_test
      (optionally cv_acc)
    """
    df = df.copy()

    # enforce order
    if "SQI" in df.columns:
        df["SQI"] = df["SQI"].astype(str)
        df["_order"] = df["SQI"].map({k: i for i, k in enumerate(SQI_ORDER)}).fillna(999).astype(int)
        df = df.sort_values("_order").drop(columns=["_order"])

    # rename to paper style
    rename = {
        "Ac_train": "Ac(train)",
        "Se_train": "Se(train)",
        "Sp_train": "Sp(train)",
        "Ac_test": "Ac(test)",
        "Se_test": "Se(test)",
        "Sp_test": "Sp(test)",
        "cv_acc": "CV(Ac)",
    }
    df = df.rename(columns=rename)

    # numeric formatting (string)
    for c in ["CV(Ac)", "Ac(train)", "Se(train)", "Sp(train)", "Ac(test)", "Se(test)", "Sp(test)"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: _fmt_float(x, fmt.metric_nd))

    # reorder cols
    cols = ["SQI", "n_features", "CV(Ac)", "Ac(train)", "Se(train)", "Sp(train)", "Ac(test)", "Se(test)", "Sp(test)"]
    df = _reorder_cols(df, cols)
    return df


def make_table6_svm(df: pd.DataFrame, fmt: PaperFormat = PAPER) -> pd.DataFrame:
    """
    Input:
      Group, Selected_SQI, n_features, cv_acc, Ac_train, Ac_test
    Output:
      Group, Selected_SQI, n_features, CV(Ac), Ac(train), Ac(test)
    """
    df = df.copy()
    if "Group" in df.columns:
        df["Group"] = df["Group"].astype(str)
        df["_order"] = df["Group"].map({k: i for i, k in enumerate(GROUP_ORDER)}).fillna(999).astype(int)
        df = df.sort_values("_order").drop(columns=["_order"])

    rename = {
        "Selected_SQI": "Selected SQIs",
        "cv_acc": "CV(Ac)",
        "Ac_train": "Ac(train)",
        "Ac_test": "Ac(test)",
    }
    df = df.rename(columns=rename)

    for c in ["CV(Ac)", "Ac(train)", "Ac(test)"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: _fmt_float(x, fmt.metric_nd))

    cols = ["Group", "Selected SQIs", "n_features", "CV(Ac)", "Ac(train)", "Ac(test)"]
    df = _reorder_cols(df, cols)
    return df


def make_table5_mlp(df: pd.DataFrame, fmt: PaperFormat = PAPER) -> pd.DataFrame:
    """
    Input (your lm_mlp_search tables):
      SQI, n_features, J_star, Ac_train, Se_train, Sp_train, Ac_test, Se_test, Sp_test, AUC_test, ...
    Output:
      SQI, n_features, J*, Ac(train), Se(train), Sp(train), Ac(test), Se(test), Sp(test), AUC(test)
    """
    df = df.copy()

    if "SQI" in df.columns:
        df["SQI"] = df["SQI"].astype(str)
        df["_order"] = df["SQI"].map({k: i for i, k in enumerate(SQI_ORDER)}).fillna(999).astype(int)
        df = df.sort_values("_order").drop(columns=["_order"])

    rename = {
        "J_star": "J*",
        "Ac_train": "Ac(train)",
        "Se_train": "Se(train)",
        "Sp_train": "Sp(train)",
        "Ac_test": "Ac(test)",
        "Se_test": "Se(test)",
        "Sp_test": "Sp(test)",
        "AUC_test": "AUC(test)",
    }
    df = df.rename(columns=rename)

    # format ints
    if "J*" in df.columns:
        df["J*"] = df["J*"].apply(lambda x: "" if pd.isna(x) else str(int(float(x))))

    # format metrics
    for c in ["Ac(train)", "Se(train)", "Sp(train)", "Ac(test)", "Se(test)", "Sp(test)"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: _fmt_float(x, fmt.metric_nd))

    if fmt.keep_auc and "AUC(test)" in df.columns:
        df["AUC(test)"] = df["AUC(test)"].apply(lambda x: _fmt_float(x, fmt.auc_nd))
    elif "AUC(test)" in df.columns:
        df = df.drop(columns=["AUC(test)"])

    cols = ["SQI", "n_features", "J*", "Ac(train)", "Se(train)", "Sp(train)", "Ac(test)", "Se(test)", "Sp(test)", "AUC(test)"]
    df = _reorder_cols(df, cols)
    return df


def make_table6_mlp(df: pd.DataFrame, fmt: PaperFormat = PAPER) -> pd.DataFrame:
    """
    Input:
      Group, Selected_SQI, n_features, J_star, Ac_train, Ac_test, AUC_test, ...
    Output:
      Group, Selected SQIs, n_features, J*, Ac(train), Ac(test), AUC(test)
    """
    df = df.copy()

    if "Group" in df.columns:
        df["Group"] = df["Group"].astype(str)
        df["_order"] = df["Group"].map({k: i for i, k in enumerate(GROUP_ORDER)}).fillna(999).astype(int)
        df = df.sort_values("_order").drop(columns=["_order"])

    rename = {
        "Selected_SQI": "Selected SQIs",
        "J_star": "J*",
        "Ac_train": "Ac(train)",
        "Ac_test": "Ac(test)",
        "AUC_test": "AUC(test)",
    }
    df = df.rename(columns=rename)

    if "J*" in df.columns:
        df["J*"] = df["J*"].apply(lambda x: "" if pd.isna(x) else str(int(float(x))))

    for c in ["Ac(train)", "Ac(test)"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: _fmt_float(x, fmt.metric_nd))

    if fmt.keep_auc and "AUC(test)" in df.columns:
        df["AUC(test)"] = df["AUC(test)"].apply(lambda x: _fmt_float(x, fmt.auc_nd))
    elif "AUC(test)" in df.columns:
        df = df.drop(columns=["AUC(test)"])

    cols = ["Group", "Selected SQIs", "n_features", "J*", "Ac(train)", "Ac(test)", "AUC(test)"]
    df = _reorder_cols(df, cols)
    return df


def make_table7_mlp(df: pd.DataFrame, fmt: PaperFormat = PAPER) -> pd.DataFrame:
    """
    Input:
      Setting, Selected_SQI, n_features, J_star, Ac_train, Se_train, Sp_train, Ac_test, Se_test, Sp_test, AUC_test
    Output:
      Selected SQIs, n_features, J*, Ac(test), Se(test), Sp(test), AUC(test)
      (and optionally train cols if you want)
    """
    df = df.copy()
    rename = {
        "Selected_SQI": "Selected SQIs",
        "J_star": "J*",
        "Ac_train": "Ac(train)",
        "Se_train": "Se(train)",
        "Sp_train": "Sp(train)",
        "Ac_test": "Ac(test)",
        "Se_test": "Se(test)",
        "Sp_test": "Sp(test)",
        "AUC_test": "AUC(test)",
    }
    df = df.rename(columns=rename)

    if "J*" in df.columns:
        df["J*"] = df["J*"].apply(lambda x: "" if pd.isna(x) else str(int(float(x))))

    for c in ["Ac(train)", "Se(train)", "Sp(train)", "Ac(test)", "Se(test)", "Sp(test)"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: _fmt_float(x, fmt.metric_nd))

    if fmt.keep_auc and "AUC(test)" in df.columns:
        df["AUC(test)"] = df["AUC(test)"].apply(lambda x: _fmt_float(x, fmt.auc_nd))
    elif "AUC(test)" in df.columns:
        df = df.drop(columns=["AUC(test)"])

    # keep it concise like paper table
    cols = ["Selected SQIs", "n_features", "J*", "Ac(test)", "Se(test)", "Sp(test)", "AUC(test)"]
    df = _reorder_cols(df, cols)
    return df


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make paper-aligned tables from SVM/MLP outputs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--artifacts_dir", type=str, default="artifacts")
    p.add_argument("--out_dir", type=str, default="artifacts/paper_tables")
    p.add_argument("--metric_nd", type=int, default=3)
    p.add_argument("--auc_nd", type=int, default=3)
    p.add_argument("--no_auc", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    seed = int(args.seed)
    root = project_root()
    artifacts = root / args.artifacts_dir
    out_dir = root / args.out_dir
    _ensure_outdir(out_dir)

    fmt = PaperFormat(
        metric_nd=int(args.metric_nd),
        auc_nd=int(args.auc_nd),
        keep_auc=(not bool(args.no_auc)),
    )

    # ---- inputs (default paths based on your pipeline) ----
    svm_dir = artifacts / "models" / "svm"
    mlp_dir = artifacts / "models" / "lm_mlp"

    svm_tables = svm_dir/"tables"
    svm_t5 = svm_tables/ f"table5_12lead_single_sqi_seed{seed}.csv"
    svm_t6 = svm_tables / f"table6_12lead_combo_sqi_seed{seed}.csv"

    mlp_tables = mlp_dir / "tables"
    mlp_t5 = mlp_tables / f"table5_mlp_12lead_single_sqi_seed{seed}.csv"
    mlp_t6 = mlp_tables / f"table6_mlp_12lead_combo_sqi_seed{seed}.csv"
    mlp_t7 = mlp_tables / f"table7_mlp_selected5_seed{seed}.csv"

    # ---- load ----
    dfs: dict[str, pd.DataFrame] = {}

    if svm_t5.exists():
        dfs["svm_t5"] = make_table5_svm(_read_csv(svm_t5), fmt=fmt)
    if svm_t6.exists():
        dfs["svm_t6"] = make_table6_svm(_read_csv(svm_t6), fmt=fmt)

    if mlp_t5.exists():
        dfs["mlp_t5"] = make_table5_mlp(_read_csv(mlp_t5), fmt=fmt)
    if mlp_t6.exists():
        dfs["mlp_t6"] = make_table6_mlp(_read_csv(mlp_t6), fmt=fmt)
    if mlp_t7.exists():
        dfs["mlp_t7"] = make_table7_mlp(_read_csv(mlp_t7), fmt=fmt)

    if not dfs:
        raise FileNotFoundError(
            "No input tables found. Run svm_tables / lm_mlp_search(tables=True) first."
        )

    # ---- save ----
    if "svm_t5" in dfs:
        p = out_dir / "paper_table5_svm.csv"
        dfs["svm_t5"].to_csv(p, index=False)
        print(f"[saved] {p}")

    if "svm_t6" in dfs:
        p = out_dir / "paper_table6_svm.csv"
        dfs["svm_t6"].to_csv(p, index=False)
        print(f"[saved] {p}")

    if "mlp_t5" in dfs:
        p = out_dir / "paper_table5_mlp.csv"
        dfs["mlp_t5"].to_csv(p, index=False)
        print(f"[saved] {p}")

    if "mlp_t6" in dfs:
        p = out_dir / "paper_table6_mlp.csv"
        dfs["mlp_t6"].to_csv(p, index=False)
        print(f"[saved] {p}")

    if "mlp_t7" in dfs:
        p = out_dir / "paper_table7_mlp.csv"
        dfs["mlp_t7"].to_csv(p, index=False)
        print(f"[saved] {p}")

    # optional: also export xlsx in one file (nice for Word copy/paste)
    xlsx_path = out_dir / "paper_tables.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        for k, df in dfs.items():
            sheet = k[:31]  # Excel sheet name limit
            df.to_excel(w, sheet_name=sheet, index=False)
    print(f"[saved] {xlsx_path}")


if __name__ == "__main__":
    main()
