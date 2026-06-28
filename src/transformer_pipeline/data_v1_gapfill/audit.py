from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .common import POLICY, protocol_dir, split_dir

EXPECTED_TYPES = ["but_native_morph", "clean_style", "original_but", "ptb_morph"]
EXPECTED_PROTOCOL = {"bad": 10530, "good": 10530, "medium": 10530}
EXPECTED_TRAIN = {"bad": 8310, "good": 8310, "medium": 8310}


def load_atlas(path):
    return pd.read_csv(path, low_memory=False)


def donor_split_problems(split: pd.DataFrame) -> int:
    original = split[split["v116_candidate_type"].astype(str).eq("original_but")]
    source_to_split = dict(zip(original["source_idx"].astype(str), original["split"].astype(str)))
    idx_to_split = dict(zip(original["idx"].astype(str), original["split"].astype(str)))
    problems = 0
    generated_train = split[split["split"].eq("train") & ~split["v116_candidate_type"].astype(str).eq("original_but")]
    for _, row in generated_train.iterrows():
        linked = None
        for col in ["v116_native_donor_id", "v116_style_donor_id", "v114_donor_source_idx"]:
            if col not in row.index or pd.isna(row[col]):
                continue
            key = str(int(float(row[col])))
            linked = source_to_split.get(key, idx_to_split.get(key, linked))
        if "v116_residual_donor_id" in row.index and pd.notna(row["v116_residual_donor_id"]):
            record = str(int(float(row["v116_residual_donor_id"])))
            mode = original.loc[original["record_id"].astype(str).eq(record), "split"].mode()
            if len(mode):
                linked = str(mode.iloc[0])
        if linked not in (None, "train"):
            problems += 1
    return problems


def raw_alias_drift(frame: pd.DataFrame) -> dict[str, float]:
    pairs = {
        "raw_rms_vs_rms": ("raw_rms", "rms"),
        "raw_ptp_vs_ptp": ("raw_ptp_p99_p01", "ptp_p99_p01"),
        "raw_diff_vs_non_qrs_diff": ("raw_diff_abs_p95", "non_qrs_diff_p95"),
    }
    out: dict[str, float] = {}
    for name, (left, right) in pairs.items():
        if left not in frame.columns or right not in frame.columns:
            out[name] = float("inf")
            continue
        delta = (pd.to_numeric(frame[left], errors="coerce") - pd.to_numeric(frame[right], errors="coerce")).abs()
        out[name] = float(delta.max(skipna=True))
    return out


def payload() -> dict[str, Any]:
    atlas = load_atlas(protocol_dir() / "original_region_atlas.csv")
    split = load_atlas(split_dir() / "original_region_atlas.csv")
    original = atlas[atlas["v116_candidate_type"].astype(str).eq("original_but")]
    out: dict[str, Any] = {
        "policy": POLICY,
        "protocol_rows": int(len(atlas)),
        "protocol_class_counts": atlas["class_name"].value_counts().sort_index().astype(int).to_dict(),
        "original_but_rows": int(len(original)),
        "original_but_class_counts": original["class_name"].value_counts().sort_index().astype(int).to_dict(),
        "train_class_counts": split.loc[split["split"].eq("train"), "class_name"].value_counts().sort_index().astype(int).to_dict(),
        "val_test_generated_rows": int(len(split[split["split"].isin(["val", "test"]) & ~split["v116_candidate_type"].astype(str).eq("original_but")])),
        "allowed_candidate_types": sorted(split.loc[split["split"].ne("unused"), "v116_candidate_type"].astype(str).unique().tolist()),
        "missing_class_rows": int(split.loc[split["split"].ne("unused"), "class_name"].isna().sum()),
        "missing_idx_rows": int(pd.to_numeric(split.loc[split["split"].ne("unused"), "idx"], errors="coerce").isna().sum()),
        "raw_alias_max_abs_delta": raw_alias_drift(atlas),
    }
    out["train_generated_donor_split_problems"] = donor_split_problems(split)
    return out


def validate(out: dict[str, Any]) -> None:
    checks = {
        "protocol_class_counts": out["protocol_class_counts"] == EXPECTED_PROTOCOL,
        "protocol_rows": out["protocol_rows"] == 31590,
        "train_class_counts": out["train_class_counts"] == EXPECTED_TRAIN,
        "val_test_generated_rows": out["val_test_generated_rows"] == 0,
        "train_generated_donor_split_problems": out["train_generated_donor_split_problems"] == 0,
        "allowed_candidate_types": out["allowed_candidate_types"] == EXPECTED_TYPES,
        "missing_class_rows": out["missing_class_rows"] == 0,
        "missing_idx_rows": out["missing_idx_rows"] == 0,
        "raw_alias_max_abs_delta": all(float(v) < 1e-4 for v in out["raw_alias_max_abs_delta"].values()),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit("audit failed: " + ", ".join(failed))


def main() -> None:
    out = payload()
    print(json.dumps(out, indent=2, sort_keys=True))
    validate(out)
