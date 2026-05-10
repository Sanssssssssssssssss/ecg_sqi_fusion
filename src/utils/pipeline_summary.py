from __future__ import annotations

from typing import Any


def format_summary_table(summary: dict[str, Any], *, title: str = "Pipeline summary") -> str:
    """Return a compact ASCII table for a pipeline run summary."""
    rows: list[list[str]] = []
    for step in summary.get("steps", []):
        meta = step.get("meta", {}) or {}
        rows.append(
            [
                str(step.get("name", "")),
                "skip" if step.get("skipped", False) else "done",
                _format_seconds(meta.get("duration_sec")),
                str(len(step.get("outputs", []) or [])),
                _format_note(meta),
            ]
        )

    headers = ["step", "status", "sec", "outputs", "note"]
    if not rows:
        rows = [["(none)", "-", "-", "0", ""]]

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    line = "  ".join("-" * width for width in widths)
    out = [title, "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))), line]
    out.extend("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))) for row in rows)
    return "\n".join(out)


def _format_seconds(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.1f}"
    return "-"


def _format_note(meta: dict[str, Any]) -> str:
    ignore = {"step", "skipped", "outputs", "duration_sec"}
    priority = (
        "dry_run",
        "rows",
        "total",
        "ok",
        "fail",
        "read_records",
        "generated_segments",
        "filtered_segments",
        "train_segments",
        "val_segments",
        "test_segments",
        "valid_rr",
        "p_mean",
        "accuracy",
        "auc",
        "test_accuracy",
        "test_auc",
    )
    parts: list[str] = []
    for key in priority:
        if key in meta:
            parts.append(f"{key}={_short_value(meta[key])}")
    if not parts:
        for key, value in meta.items():
            if key in ignore:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                parts.append(f"{key}={_short_value(value)}")
            if len(parts) >= 2:
                break
    return ", ".join(parts[:3])


def _short_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    text = str(value)
    return text if len(text) <= 32 else text[:29] + "..."
