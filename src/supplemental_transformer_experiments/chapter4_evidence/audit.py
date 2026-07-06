from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.transformer_pipeline.data_v1_gapfill.common import protocol_dir, split_dir
from src.supplemental_transformer_experiments.sqi12_gapfill import run as sqi12

from .common import Paths, dry, ensure_dirs, write_json


def _seta_audit(paths: Paths) -> dict[str, Any]:
    sp = sqi12.Paths(paths.seta)
    out = sqi12.audit_dict(sp)
    expected_manifest = {"acceptable": 773, "unacceptable": 225}
    if out["manifest_rows"] != 998 or out["manifest_counts"] != expected_manifest:
        raise AssertionError(f"Set-A manifest mismatch: {out['manifest_counts']}")
    expected_split = {
        "train_acceptable": 541,
        "train_unacceptable": 158,
        "val_acceptable": 116,
        "val_unacceptable": 34,
        "test_acceptable": 116,
        "test_unacceptable": 33,
    }
    if out["split_counts"] != expected_split:
        raise AssertionError(f"Set-A split mismatch: {out['split_counts']}")
    if out["train_counts"] != {"0": 541, "1": 541}:
        raise AssertionError(f"Set-A train balance mismatch: {out['train_counts']}")
    if int(out["val_test_generated_rows"]) != 0:
        raise AssertionError("Set-A val/test contains generated rows")
    if int(out["train_generated_donor_split_problems"]) != 0:
        raise AssertionError("Set-A donor leakage detected")
    return out


def _but_audit() -> dict[str, Any]:
    protocol = pd.read_csv(protocol_dir() / "metadata.csv", low_memory=False)
    split = pd.read_csv(split_dir() / "original_region_atlas.csv", low_memory=False)
    signals = np.load(protocol_dir() / "signals.npz", allow_pickle=True)["X"]
    out = {
        "protocol_rows": int(len(protocol)),
        "signals_shape": list(signals.shape),
        "protocol_class_counts": {str(k): int(v) for k, v in protocol["class_name"].value_counts().sort_index().items()},
        "original_but_rows": int(protocol["v116_candidate_type"].eq("original_but").sum()),
        "original_but_class_counts": {
            str(k): int(v)
            for k, v in protocol.loc[protocol["v116_candidate_type"].eq("original_but"), "class_name"].value_counts().sort_index().items()
        },
        "split_counts": {
            f"{a}_{b}": int(v) for (a, b), v in split.groupby(["split", "class_name"]).size().items()
        },
        "train_class_counts": {
            str(k): int(v) for k, v in split.loc[split["split"].eq("train"), "class_name"].value_counts().sort_index().items()
        },
        "val_test_generated_rows": int(
            split.loc[split["split"].isin(["val", "test"]) & ~split["v116_candidate_type"].astype(str).eq("original_but")].shape[0]
        ),
        "allowed_candidate_types": sorted(split["v116_candidate_type"].dropna().astype(str).unique().tolist()),
        "protocol_path": str(protocol_dir()),
        "split_path": str(split_dir()),
    }
    if out["protocol_rows"] != 31590:
        raise AssertionError(f"BUT protocol row mismatch: {out['protocol_rows']}")
    if out["protocol_class_counts"] != {"bad": 10530, "good": 10530, "medium": 10530}:
        raise AssertionError(f"BUT class balance mismatch: {out['protocol_class_counts']}")
    if out["train_class_counts"] != {"bad": 8310, "good": 8310, "medium": 8310}:
        raise AssertionError(f"BUT train split mismatch: {out['train_class_counts']}")
    if out["val_test_generated_rows"] != 0:
        raise AssertionError("BUT val/test contains generated rows")
    return out


def run(paths: Paths, *, execute: bool, scope: str = "all") -> dict[str, Any]:
    if not execute:
        dry("audit", paths)
        return {"step": "audit", "skipped": True, "scope": scope}
    ensure_dirs(paths)
    if scope not in {"all", "seta", "but"}:
        raise ValueError(f"unknown audit scope: {scope}")
    out: dict[str, Any] = {}
    if scope in {"all", "seta"}:
        out["seta"] = _seta_audit(paths)
    if scope in {"all", "but"}:
        out["but"] = _but_audit()
    out["scope"] = scope
    write_json(paths.audit_json, out)
    print(paths.audit_json)
    return {"step": "audit", "skipped": False, "scope": scope, "outputs": out}
