from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.paths import project_root


METRIC_TOL = 0.005


def main() -> None:
    args = parse_args()
    root = project_root()
    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_absolute():
        artifacts_dir = root / artifacts_dir

    if args.compare:
        before = _read_json(Path(args.compare[0]))
        after = _read_json(Path(args.compare[1]))
        result = compare_summaries(before, after)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["failed"]:
            raise SystemExit(1)
        return

    summary = collect_summary(root, artifacts_dir, seed=args.seed)
    if args.write:
        out = Path(args.write)
        if not out.is_absolute():
            out = root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(out.relative_to(root).as_posix())
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SQI pipeline artifacts.")
    parser.add_argument("--artifacts_dir", default="outputs/sqi")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write", default="")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    return parser.parse_args()


def collect_summary(root: Path, artifacts_dir: Path, *, seed: int = 0) -> dict[str, Any]:
    split_path = artifacts_dir / "splits" / f"split_seta_seed{seed}_balanced.csv"
    split = pd.read_csv(split_path)
    features_dir = artifacts_dir / "features"

    run_summary = _read_json(artifacts_dir / "config" / f"run_summary_seed{seed}.json")
    absolute_outputs: list[str] = []
    missing_outputs: list[str] = []
    for step in run_summary.get("steps", []):
        for out in step.get("outputs", []):
            text = str(out).replace("\\", "/")
            if Path(text).is_absolute() or ":" in text[:3]:
                absolute_outputs.append(str(out))
            path = Path(text)
            if not path.is_absolute():
                path = root / path
            if not path.exists():
                missing_outputs.append(str(out))

    return {
        "seed": seed,
        "split": {
            "path": _rel(split_path, root),
            "shape": list(split.shape),
            "split_counts": _counts(split["split"]),
            "label_counts": _counts(split["y"]),
            "columns": list(split.columns),
        },
        "features": {
            "record84": _frame_summary(features_dir / "record84.parquet", root),
            "record84_norm": _frame_summary(features_dir / "record84_norm.parquet", root),
            "lead7": _frame_summary(features_dir / "lead7.parquet", root),
        },
        "metrics": {
            "lm_mlp": _read_json(artifacts_dir / "models" / "lm_mlp" / f"lm_mlp_test_metrics_seed{seed}.json"),
            "logreg": _read_json(artifacts_dir / "models" / "baselines" / "logreg" / f"logreg_test_metrics_seed{seed}.json"),
            "gnb": _read_json(artifacts_dir / "models" / "baselines" / "gnb" / f"gnb_test_metrics_seed{seed}.json"),
        },
        "svm_tables": {
            "table5_shape": list(pd.read_csv(artifacts_dir / "models" / "svm" / f"table5_12lead_single_sqi_seed{seed}.csv").shape),
            "table6_shape": list(pd.read_csv(artifacts_dir / "models" / "svm" / f"table6_12lead_combo_sqi_seed{seed}.csv").shape),
        },
        "run_summary": {
            "path": _rel(artifacts_dir / "config" / f"run_summary_seed{seed}.json", root),
            "step_count": len(run_summary.get("steps", [])),
            "absolute_outputs": absolute_outputs,
            "missing_outputs": missing_outputs,
        },
    }


def compare_summaries(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("split_shape", before["split"]["shape"] == after["split"]["shape"], after["split"]["shape"])
    add("split_counts", before["split"]["split_counts"] == after["split"]["split_counts"], after["split"]["split_counts"])
    add("label_counts", before["split"]["label_counts"] == after["split"]["label_counts"], after["split"]["label_counts"])

    for key in ["record84", "record84_norm", "lead7"]:
        add(f"{key}_shape", before["features"][key]["shape"] == after["features"][key]["shape"], after["features"][key]["shape"])
        add(f"{key}_columns", before["features"][key]["columns"] == after["features"][key]["columns"])
        add(
            f"{key}_numeric_digest",
            before["features"][key]["numeric_sha256_round8"] == after["features"][key]["numeric_sha256_round8"],
        )

    _compare_metric(checks, before, after, "lm_mlp", ("test_metrics_fixed",), ["acc", "auc", "se", "sp"])
    _compare_metric(checks, before, after, "logreg", ("test_metrics_fixed",), ["acc", "auc", "se", "sp"])
    _compare_metric(checks, before, after, "gnb", ("test_metrics_at_val_threshold",), ["acc", "auc", "se", "sp"])

    add("lm_mlp_confusion_matrix", _nested(before, "metrics", "lm_mlp", "test_metrics_fixed", "confusion_matrix") == _nested(after, "metrics", "lm_mlp", "test_metrics_fixed", "confusion_matrix"))
    add("logreg_confusion_matrix", _nested(before, "metrics", "logreg", "test_metrics_fixed", "confusion_matrix") == _nested(after, "metrics", "logreg", "test_metrics_fixed", "confusion_matrix"))
    add("run_summary_no_absolute_outputs", len(after["run_summary"]["absolute_outputs"]) == 0, len(after["run_summary"]["absolute_outputs"]))
    add("run_summary_no_missing_outputs", len(after["run_summary"]["missing_outputs"]) == 0, after["run_summary"]["missing_outputs"])

    failed = [item for item in checks if not item["ok"]]
    return {"failed": failed, "checks": checks}


def _compare_metric(checks: list[dict[str, Any]], before: dict[str, Any], after: dict[str, Any], model: str, path: tuple[str, ...], names: list[str]) -> None:
    for name in names:
        b = _nested(before, "metrics", model, *path, name)
        a = _nested(after, "metrics", model, *path, name)
        ok = b is not None and a is not None and abs(float(b) - float(a)) <= METRIC_TOL
        checks.append({"name": f"{model}_{name}", "ok": ok, "detail": {"before": b, "after": a}})


def _nested(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _frame_summary(path: Path, root: Path) -> dict[str, Any]:
    df = pd.read_parquet(path)
    return {
        "path": _rel(path, root),
        "shape": list(df.shape),
        "columns": list(df.columns),
        "numeric_sha256_round8": _numeric_digest(df),
    }


def _numeric_digest(df: pd.DataFrame) -> str:
    numeric = df.select_dtypes(include=[np.number]).round(8)
    return hashlib.sha256(numeric.to_csv(index=False).encode("utf-8")).hexdigest()


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(series.value_counts(dropna=False).to_dict().items(), key=lambda item: str(item[0]))}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
