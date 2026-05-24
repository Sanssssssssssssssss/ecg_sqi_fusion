from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]


def load_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def recalls(cm: list[list[int]]) -> tuple[float, float, float]:
    out: list[float] = []
    for i, row in enumerate(cm):
        out.append(float(row[i]) / float(max(1, sum(row))))
    return out[0], out[1], out[2]


def fmt(v: float | None) -> str:
    if v is None:
        return ""
    return f"{v:.4f}"


def row(label: str, path: Path) -> tuple[str, float | None, str]:
    rep = load_report(path)
    if rep is None:
        return f"| {label} | missing |  |  |  | `{path}` |", None, label
    cm = rep["confusion_matrix_3x3"]
    good, medium, bad = recalls(cm)
    acc = float(rep["acc"])
    return f"| {label} | {acc:.4f} | {good:.4f} | {medium:.4f} | {bad:.4f} | `{cm}` |", acc, label


def main() -> None:
    runs = [
        (
            "D1 E3.9a reference",
            ROOT / "outputs/transformer_e39a_smooth_morph_triplet/models/e39_d1_smooth_morph_cls_raw/test_report.json",
        ),
        (
            "E3.10 M0 baseline",
            ROOT / "outputs/transformer_e310_smooth_morph_mild_snr/models/e310_smooth_morph_mild_snr_cls_raw/test_report.json",
        ),
        (
            "E3.10 M1 D1 warm-start",
            ROOT / "outputs/transformer_e310_smooth_morph_mild_snr/models/e310_m1_d1warm_lr3e5/test_report.json",
        ),
        (
            "E3.10 M2 warm-start + SNR head",
            ROOT / "outputs/transformer_e310_smooth_morph_mild_snr/models/e310_m2_d1warm_snr005/test_report.json",
        ),
        (
            "E3.10 M3 warm-start + SNR head + low denoise",
            ROOT / "outputs/transformer_e310_smooth_morph_mild_snr/models/e310_m3_d1warm_snr005_lowden/test_report.json",
        ),
        (
            "E3.10 M4 M3 + noise type head",
            ROOT / "outputs/transformer_e310_smooth_morph_mild_snr/models/e310_m4_d1warm_snr005_lowden_ntype005/test_report.json",
        ),
        (
            "E3.11 M0 baseline",
            ROOT / "outputs/transformer_e311_visual_gap/models/e311_visual_gap_cls_raw/test_report.json",
        ),
        (
            "E3.11 previous best tune",
            ROOT / "outputs/transformer_e311_visual_gap/models/e311_visual_gap_cls_raw_reg_ls003/test_report.json",
        ),
        (
            "E3.11 M1 D1 warm-start",
            ROOT / "outputs/transformer_e311_visual_gap/models/e311_m1_d1warm_lr3e5/test_report.json",
        ),
        (
            "E3.11 M2 warm-start + SNR head",
            ROOT / "outputs/transformer_e311_visual_gap/models/e311_m2_d1warm_snr005/test_report.json",
        ),
        (
            "E3.11 M3 warm-start + SNR head + low denoise",
            ROOT / "outputs/transformer_e311_visual_gap/models/e311_m3_d1warm_snr005_lowden/test_report.json",
        ),
        (
            "E3.11 M4 M3 + noise type head",
            ROOT / "outputs/transformer_e311_visual_gap/models/e311_m4_d1warm_snr005_lowden_ntype005/test_report.json",
        ),
    ]

    lines = [
        "# E3.10/E3.11 SNR Warm-Start Sweep",
        "",
        "Goal: test whether D1 warm-start plus explicit SNR learning can rescue the visual-SNR datasets without changing the model architecture.",
        "",
        "| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Confusion Matrix |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    best_e310: tuple[str, float] | None = None
    best_e311: tuple[str, float] | None = None
    for label, path in runs:
        line, acc, name = row(label, path)
        lines.append(line)
        if acc is None:
            continue
        if label.startswith("E3.10") and "baseline" not in label.lower():
            if best_e310 is None or acc > best_e310[1]:
                best_e310 = (name, acc)
        if label.startswith("E3.11") and "baseline" not in label.lower() and "previous" not in label.lower():
            if best_e311 is None or acc > best_e311[1]:
                best_e311 = (name, acc)

    lines.extend(
        [
            "",
            "## Best New Runs",
            "",
            f"- E3.10 best new run: `{best_e310[0]}` = `{best_e310[1]:.4f}`" if best_e310 else "- E3.10 best new run: pending",
            f"- E3.11 best new run: `{best_e311[0]}` = `{best_e311[1]:.4f}`" if best_e311 else "- E3.11 best new run: pending",
            "",
            "Success criteria:",
            "",
            "- E3.11 `>=0.90`: rescued enough for further analysis",
            "- E3.11 `>=0.93`: usable visual benchmark",
            "- E3.11 `>=0.94`: strong visual benchmark",
            "- E3.10 `>=0.94`: candidate visual version if E3.11 remains too severe",
        ]
    )

    out = ROOT / "reports/transformer_e310_e311_snr_warmstart_sweep_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
