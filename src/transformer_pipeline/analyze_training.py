from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.utils.paths import project_root


def main() -> None:
    args = parse_args()
    root = project_root()
    model_dirs = [resolve_path(root, p) for p in args.model_dir]
    if not model_dirs:
        base = resolve_path(root, args.artifact_dir) / "models"
        model_dirs = sorted(p for p in base.iterdir() if (p / "probe_summary.json").exists())

    rows = [summarize_dir(root, path) for path in model_dirs]
    rows = [row for row in rows if row]
    rows.sort(key=lambda row: (row.get("test_acc") or -1.0), reverse=True)

    if args.write:
        out = resolve_path(root, args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(rel(out, root))

    print_table(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize transformer tuning runs.")
    parser.add_argument("--artifact_dir", default="outputs/transformer")
    parser.add_argument("--model_dir", action="append", default=[])
    parser.add_argument("--write", default="")
    return parser.parse_args()


def summarize_dir(root: Path, model_dir: Path) -> dict[str, Any]:
    probe_path = model_dir / "probe_summary.json"
    test_path = model_dir / "test_report.json"
    if probe_path.exists():
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
    elif test_path.exists():
        probe = {"test_acc": json.loads(test_path.read_text(encoding="utf-8")).get("acc")}
    else:
        return {}

    hp = probe.get("hyperparams", {})
    best_acc = probe.get("best_val_acc_epoch", {})
    best_loss = probe.get("best_val_loss_epoch", {})
    return {
        "model_dir": rel(model_dir, root),
        "test_acc": probe.get("test_acc"),
        "best_val_acc": best_acc.get("val_acc"),
        "best_val_acc_epoch": best_acc.get("epoch"),
        "best_val_loss": best_loss.get("val_loss"),
        "best_val_loss_epoch": best_loss.get("epoch"),
        "dropout": hp.get("dropout"),
        "lr": hp.get("lr"),
        "weight_decay": hp.get("weight_decay"),
        "cls_pool": hp.get("cls_pool"),
        "lambda_den": hp.get("lambda_den"),
        "bad_den_w_max": hp.get("bad_den_w_max"),
        "label_smoothing": hp.get("label_smoothing"),
        "class_weight_medium": hp.get("class_weight_medium"),
        "uncertainty_mode": hp.get("uncertainty_mode"),
        "e_cls": hp.get("e_cls"),
        "e_denoise": hp.get("e_denoise"),
        "e_level": hp.get("e_level"),
        "e_uncert": hp.get("e_uncert"),
        "select_best_by": hp.get("select_best_by"),
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No transformer tuning runs found.")
        return
    headers = [
        "model_dir",
        "test",
        "best_val",
        "epoch",
        "drop",
        "lr",
        "wd",
        "pool",
        "lden",
        "badw",
        "ls",
        "mw",
        "umode",
        "sched",
    ]
    table: list[list[str]] = []
    for row in rows:
        schedule = f"{row.get('e_cls')}/{row.get('e_denoise')}/{row.get('e_level')}/{row.get('e_uncert')}"
        table.append([
            str(row.get("model_dir")),
            fmt(row.get("test_acc")),
            fmt(row.get("best_val_acc")),
            str(row.get("best_val_acc_epoch")),
            fmt(row.get("dropout")),
            fmt(row.get("lr")),
            fmt(row.get("weight_decay")),
            str(row.get("cls_pool") or ""),
            fmt(row.get("lambda_den")),
            fmt(row.get("bad_den_w_max")),
            fmt(row.get("label_smoothing")),
            fmt(row.get("class_weight_medium")),
            str(row.get("uncertainty_mode") or ""),
            schedule,
        ])
    widths = [len(h) for h in headers]
    for row in table:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in table:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
