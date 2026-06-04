"""BUT 10s fatal-OR synthetic rule grid.

This pass is intentionally narrower than the broad morphology-guided grid.  It
tests a specific protocol hypothesis from visual BUT review:

* good = all critical dimensions look usable;
* bad = any one fatal dimension fails hard;
* medium = QRS is still usable, but at least one detail dimension is unreliable.

The implementation reuses the existing experiment-only generator, Uformer
trainer, 10s BUT evaluator, and feature audit.  It does not touch the mainline
checkpoint or src/sqi_pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks import but_morphology_guided_grid_10s as mg
from src.transformer_pipeline.external_benchmarks.but_adaptation_14h import ROOT, load_ptb_artifact
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_tuning import (
    DEFAULT_SOURCE_ARTIFACT,
    evaluate_checkpoint_10s,
    now_iso,
    plot_synthetic_gallery,
    read_json,
    update_state,
)
from src.transformer_pipeline.external_benchmarks.but_generator_morph_sweet_grid import apply_morph_spec
from src.transformer_pipeline.external_benchmarks.but_protocol_adaptation import (
    DEFAULT_OUT_ROOT as PROTOCOL_OUT_ROOT,
    DEFAULT_REPORT_ROOT as PROTOCOL_REPORT_ROOT,
)


RUN_TAG = "e311_but_fatal_or_logic_grid_10s_2026_06_04"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
STATE_NAME = "fatal_or_logic_state.json"
SUMMARY_NAME = "fatal_or_logic_summary.jsonl"

B10_ANCHOR = {
    "acc": 0.7735,
    "balanced_acc": 0.8045,
    "macro_f1": 0.7238,
    "recall_good": 0.824,
    "recall_medium": 0.724,
    "recall_bad": 0.866,
}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_entry(family: str, name: str, changes: dict[str, Any], class_weight: str, note: str) -> dict[str, Any]:
    spec = mg.lrg.base_spec()
    spec.update(changes)
    spec["id"] = f"{family}_{name}"
    spec["family"] = family
    return {"spec": spec, "class_weight": class_weight, "note": note}


def fatal_or_entries() -> list[dict[str, Any]]:
    """Focused rules around h_bad_rescue_05 and the medium independent-cluster analysis."""

    entries: list[dict[str, Any]] = []

    # A: h_bad_rescue_05 neighborhood.  This was the first clear sweet spot:
    # better macro/acc than b10, but bad recall slightly low.  We keep its
    # medium shell and vary the fatal bad axes one at a time.
    for i, (pseudo, confuse, inv, contact, flat, step, qattn, cw) in enumerate(
        [
            (0.58, 0.56, 0.30, 0.20, 0.00, 0.44, 0.75, "1.08,1.54,1.92"),
            (0.64, 0.62, 0.34, 0.18, 0.00, 0.46, 0.76, "1.08,1.52,1.96"),
            (0.54, 0.54, 0.32, 0.28, 0.06, 0.40, 0.74, "1.10,1.56,1.92"),
            (0.48, 0.50, 0.38, 0.36, 0.10, 0.34, 0.78, "1.10,1.58,1.90"),
            (0.68, 0.66, 0.28, 0.14, 0.00, 0.50, 0.72, "1.06,1.50,1.98"),
            (0.52, 0.46, 0.44, 0.30, 0.04, 0.52, 0.80, "1.10,1.56,1.94"),
        ],
        start=1,
    ):
        entries.append(
            make_entry(
                "fatal_or_qrs_confuse",
                f"{i:02d}",
                {
                    "residual_scale_good": 0.74,
                    "residual_scale_medium": 1.03,
                    "medium_event_prob": 0.58,
                    "medium_pseudo_peak_rate": 0.06,
                    "medium_inverse_peak_rate": 0.12,
                    "medium_step_rate": 0.12,
                    "medium_ramp_rate": 0.10,
                    "medium_contact_rate": 0.03,
                    "medium_qrs_preserve": 0.997,
                    "bad_pseudo_peak_rate": pseudo,
                    "bad_qrs_confuse_rate": confuse,
                    "bad_inverse_peak_rate": inv,
                    "contact_loss_bad": contact,
                    "flatline_bad": flat,
                    "baseline_step_bad": step,
                    "qrs_attn_bad": qattn,
                },
                cw,
                "Bad as fatal QRS-confusion OR, with QRS-visible detail-unreliable medium.",
            )
        )

    # B: single fatal contact/flat axis.  BUT bad includes flat/contact cases,
    # but previous flatline-heavy rules crushed medium.  Here the medium shell
    # stays independent and the bad flat/contact axis is not combined with every
    # other bad artifact at maximum strength.
    for i, (contact, flat, step, pseudo, confuse, qattn, cw) in enumerate(
        [
            (0.42, 0.18, 0.26, 0.34, 0.30, 0.76, "1.10,1.56,1.92"),
            (0.50, 0.24, 0.24, 0.30, 0.26, 0.80, "1.10,1.58,1.92"),
            (0.56, 0.12, 0.34, 0.38, 0.34, 0.74, "1.12,1.56,1.90"),
            (0.36, 0.28, 0.38, 0.42, 0.38, 0.78, "1.08,1.58,1.94"),
        ],
        start=1,
    ):
        entries.append(
            make_entry(
                "fatal_or_contact",
                f"{i:02d}",
                {
                    "residual_scale_good": 0.76,
                    "residual_scale_medium": 1.04,
                    "medium_event_prob": 0.62,
                    "medium_pseudo_peak_rate": 0.07,
                    "medium_inverse_peak_rate": 0.14,
                    "medium_step_rate": 0.14,
                    "medium_ramp_rate": 0.12,
                    "medium_contact_rate": 0.035,
                    "medium_qrs_preserve": 0.996,
                    "bad_pseudo_peak_rate": pseudo,
                    "bad_qrs_confuse_rate": confuse,
                    "bad_inverse_peak_rate": 0.24,
                    "contact_loss_bad": contact,
                    "flatline_bad": flat,
                    "baseline_step_bad": step,
                    "qrs_attn_bad": qattn,
                },
                cw,
                "Bad as contact/flat fatal OR, not all-bad-at-once.",
            )
        )

    # C: medium independent cluster.  It should not be a midpoint to bad: QRS
    # stays highly preserved, but local derivative/baseline/P-T-ST features move.
    for i, (event, step, ramp, inv, contact, bad_pseudo, bad_contact, cw) in enumerate(
        [
            (0.66, 0.16, 0.14, 0.14, 0.04, 0.54, 0.22, "1.08,1.62,1.88"),
            (0.72, 0.20, 0.18, 0.12, 0.05, 0.52, 0.24, "1.10,1.64,1.88"),
            (0.78, 0.24, 0.22, 0.16, 0.06, 0.50, 0.20, "1.12,1.66,1.86"),
            (0.70, 0.28, 0.18, 0.20, 0.04, 0.56, 0.18, "1.10,1.64,1.90"),
        ],
        start=1,
    ):
        entries.append(
            make_entry(
                "medium_independent",
                f"{i:02d}",
                {
                    "residual_scale_good": 0.78,
                    "residual_scale_medium": 1.06,
                    "medium_event_prob": event,
                    "medium_pseudo_peak_rate": 0.08,
                    "medium_inverse_peak_rate": inv,
                    "medium_step_rate": step,
                    "medium_ramp_rate": ramp,
                    "medium_contact_rate": contact,
                    "medium_qrs_preserve": 0.997,
                    "bad_pseudo_peak_rate": bad_pseudo,
                    "bad_qrs_confuse_rate": 0.48,
                    "bad_inverse_peak_rate": 0.28,
                    "contact_loss_bad": bad_contact,
                    "flatline_bad": 0.04,
                    "baseline_step_bad": 0.42,
                    "qrs_attn_bad": 0.74,
                },
                cw,
                "Medium as independent detail-unreliable cluster with QRS preserved.",
            )
        )

    # D: explicit AND-good pressure.  Make good less overlapped with artifacts
    # while letting medium absorb imperfect-but-usable signals.
    for i, (good_scale, med_event, med_contact, bad_pseudo, bad_confuse, cw) in enumerate(
        [
            (0.68, 0.62, 0.04, 0.58, 0.56, "1.02,1.62,1.92"),
            (0.70, 0.68, 0.04, 0.56, 0.54, "1.04,1.64,1.92"),
            (0.72, 0.74, 0.05, 0.54, 0.52, "1.06,1.66,1.90"),
            (0.74, 0.70, 0.06, 0.60, 0.58, "1.08,1.64,1.94"),
        ],
        start=1,
    ):
        entries.append(
            make_entry(
                "good_and_medium_shell",
                f"{i:02d}",
                {
                    "residual_scale_good": good_scale,
                    "residual_scale_medium": 1.05,
                    "medium_event_prob": med_event,
                    "medium_pseudo_peak_rate": 0.08,
                    "medium_inverse_peak_rate": 0.14,
                    "medium_step_rate": 0.16,
                    "medium_ramp_rate": 0.14,
                    "medium_contact_rate": med_contact,
                    "medium_qrs_preserve": 0.996,
                    "bad_pseudo_peak_rate": bad_pseudo,
                    "bad_qrs_confuse_rate": bad_confuse,
                    "bad_inverse_peak_rate": 0.30,
                    "contact_loss_bad": 0.24,
                    "flatline_bad": 0.04,
                    "baseline_step_bad": 0.44,
                    "qrs_attn_bad": 0.76,
                },
                cw,
                "Good as all-critical-dimensions-good; medium absorbs imperfect usable cases.",
            )
        )

    return entries


def prepare_variant(args: argparse.Namespace, entry: dict[str, Any]) -> Path:
    spec = entry["spec"]
    variant_dir = Path(args.out_root) / "synthetic_variants" / str(spec["id"])
    labels_path = variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv"
    if not labels_path.exists():
        ptb = load_ptb_artifact(Path(args.source_artifact_dir))
        ptb["source"] = str(args.source_artifact_dir)
        payload = apply_morph_spec(ptb, spec, variant_dir, seed=int(args.seed))
        plot_synthetic_gallery(
            ptb["clean"],
            payload["X_noisy"],
            payload["labels"].reset_index(drop=True),
            variant_dir / "visuals" / "synthetic_gallery.png",
            str(spec["id"]),
        )
    # Reuse the morphology-guided feature audit so every rule is checked before
    # training, but keep the run/report namespace separate.
    entry["_feature_audit"] = mg.audit_variant(args, variant_dir, entry)
    return variant_dir


def train_one(args: argparse.Namespace, entry: dict[str, Any], mode: str) -> dict[str, Any]:
    spec = entry["spec"]
    spec_id = str(spec["id"])
    variant_dir = prepare_variant(args, entry)
    run_dir = Path(args.out_root) / "runs" / mode / spec_id
    log_dir = Path(args.out_root) / "logs" / mode
    log_dir.mkdir(parents=True, exist_ok=True)
    is_quick = mode == "quick"
    epochs1 = args.quick_epochs_stage1 if is_quick else args.full_epochs_stage1
    epochs2 = args.quick_epochs_stage2 if is_quick else args.full_epochs_stage2
    cmd = [
        sys.executable,
        "-m",
        "src.transformer_pipeline.train_uformer_mainline",
        "--stage",
        "all",
        "--source_artifact_dir",
        str(variant_dir),
        "--output_dir",
        str(run_dir),
        "--epochs_stage1",
        str(epochs1),
        "--epochs_stage2",
        str(epochs2),
        "--batch_size_stage1",
        str(args.batch_size_stage1),
        "--batch_size_stage2",
        str(args.batch_size_stage2),
        "--lambda_den_stage2",
        str(args.lambda_den_stage2),
        "--lambda_cls",
        str(args.lambda_cls),
        "--class_weight",
        str(entry["class_weight"]),
        "--seed",
        str(args.seed),
    ]
    start = time.perf_counter()
    with (log_dir / f"{spec_id}.stdout.txt").open("w", encoding="utf-8") as out, (
        log_dir / f"{spec_id}.stderr.txt"
    ).open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=out, stderr=err, text=True)
    payload: dict[str, Any] = {
        "spec": spec,
        "class_weight": entry["class_weight"],
        "note": entry["note"],
        "mode": mode,
        "seed": int(args.seed),
        "returncode": int(proc.returncode),
        "elapsed_sec": float(time.perf_counter() - start),
        "run_dir": str(run_dir),
        "variant_dir": str(variant_dir),
        "feature_audit": entry.get("_feature_audit", {}),
    }
    summary_path = run_dir / "mainline_summary.json"
    if proc.returncode == 0 and (run_dir / "ckpt_best.pt").exists() and summary_path.exists():
        summary = read_json(summary_path)
        payload["ptb_test_report"] = summary.get("stage2", {}).get("test_report", {})
        payload["ptb_denoise_metrics"] = summary.get("stage2", {}).get("denoise_metrics", {})
        payload["but_10s_eval"] = evaluate_checkpoint_10s(args, run_dir / "ckpt_best.pt", run_dir)
    else:
        payload["status"] = "failed"
    write_json(run_dir / "fatal_or_logic_run_summary.json", payload)
    write_json(run_dir / "fatal_or_logic_metric_row.json", mg.metric_row(payload))
    mg.write_visual_review(run_dir, payload)
    append_jsonl(Path(args.out_root) / SUMMARY_NAME, payload)
    return payload


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def result_key(row: dict[str, Any], args: argparse.Namespace | None = None) -> tuple[float, ...]:
    metrics = mg.row_metrics(row)
    good, medium, bad = mg.rec_from_row(row)
    valid = 1.0
    if args is not None and (metrics["ptb_acc"] < float(args.min_ptb_acc) or metrics["ptb_bad"] < float(args.min_ptb_bad)):
        valid = 0.0
    # Primary goal: improve the medium/bad coexistence without letting bad
    # precision vanish.  Macro-F1 is first, then balanced, then the lower of
    # medium/bad recall.
    return (
        valid,
        float(metrics["macro_f1"]),
        float(metrics["balanced_acc"]),
        min(float(medium), float(bad)),
        float(bad),
        float(medium),
        float(good),
    )


def select_full_entries(args: argparse.Namespace, top_n: int) -> list[dict[str, Any]]:
    rows = [r for r in load_rows(Path(args.out_root) / SUMMARY_NAME) if r.get("mode") == "quick" and int(r.get("returncode", 1)) == 0]
    ranked = sorted(rows, key=lambda r: result_key(r, args), reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ranked:
        spec = row.get("spec", {})
        spec_id = str(spec.get("id", ""))
        if not spec_id or spec_id in seen:
            continue
        selected.append({"spec": spec, "class_weight": row.get("class_weight", "1,1.56,1.90"), "note": row.get("note", "")})
        seen.add(spec_id)
        if len(selected) >= top_n:
            break
    return selected


def write_report(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    rows = load_rows(out_root / SUMMARY_NAME)
    ranked = sorted([r for r in rows if int(r.get("returncode", 1)) == 0], key=lambda r: result_key(r, args), reverse=True)
    write_json(report_root / "fatal_or_logic_rows.json", {"rows": rows, "ranked": ranked[:50], "anchor": B10_ANCHOR})

    lines = [
        "# BUT 10s Fatal-OR Logic Grid",
        "",
        "Hypothesis: BUT bad behaves like OR(any fatal usability dimension fails), good behaves like AND(all critical dimensions are usable), and medium is an independent QRS-usable/detail-unreliable cluster.",
        "",
        f"Anchor b10: acc {B10_ANCHOR['acc']:.4f}, balanced {B10_ANCHOR['balanced_acc']:.4f}, macro-F1 {B10_ANCHOR['macro_f1']:.4f}, recalls {B10_ANCHOR['recall_good']:.3f}/{B10_ANCHOR['recall_medium']:.3f}/{B10_ANCHOR['recall_bad']:.3f}.",
        "",
        "| rank | mode | spec | family | cw | BUT acc | bal | macro-F1 | recalls good/medium/bad | min med/bad | PTB acc | PTB bad | denoise | feature score | note |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    metric_rows: list[dict[str, Any]] = []
    for i, row in enumerate(ranked, start=1):
        m = mg.row_metrics(row)
        good, medium, bad = mg.rec_from_row(row)
        audit = row.get("feature_audit", {}) or {}
        spec = row.get("spec", {}) or {}
        metric_row = {
            "rank": i,
            "mode": row.get("mode"),
            "spec_id": spec.get("id"),
            "family": spec.get("family"),
            "class_weight": row.get("class_weight"),
            "but_acc": m["acc"],
            "but_balanced_acc": m["balanced_acc"],
            "but_macro_f1": m["macro_f1"],
            "recall_good": good,
            "recall_medium": medium,
            "recall_bad": bad,
            "min_medium_bad": min(medium, bad),
            "ptb_acc": m["ptb_acc"],
            "ptb_bad": m["ptb_bad"],
            "denoise_score": m["denoise_score"],
            "feature_target_score": float(audit.get("feature_target_score", 0.0)),
            "note": row.get("note", ""),
            "run_dir": row.get("run_dir", ""),
        }
        metric_rows.append(metric_row)
        lines.append(
            f"| {i} | {row.get('mode')} | {spec.get('id')} | {spec.get('family')} | {row.get('class_weight')} | "
            f"{m['acc']:.4f} | {m['balanced_acc']:.4f} | {m['macro_f1']:.4f} | "
            f"{good:.3f}/{medium:.3f}/{bad:.3f} | {min(medium, bad):.3f} | "
            f"{m['ptb_acc']:.4f} | {m['ptb_bad']:.4f} | {m['denoise_score']:.3f} | "
            f"{float(audit.get('feature_target_score', 0.0)):.3f} | {row.get('note', '')} |"
        )
    (report_root / "fatal_or_logic_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if metric_rows:
        import pandas as pd

        pd.DataFrame(metric_rows).to_csv(report_root / "fatal_or_logic_metric_join.csv", index=False)


def run_quick(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    state_path = out_root / STATE_NAME
    all_entries = fatal_or_entries()[: int(args.max_runs)]
    write_json(out_root / "fatal_or_logic_specs.json", {"entries": all_entries})
    completed = {str(r.get("spec", {}).get("id", "")) for r in load_rows(out_root / SUMMARY_NAME) if r.get("mode") == "quick"}
    for i, entry in enumerate(all_entries, start=1):
        spec_id = str(entry["spec"]["id"])
        if spec_id in completed:
            continue
        update_state(state_path, status="quick_training", stage="quick_train", current=spec_id, index=i, total=len(all_entries), updated_at=now_iso())
        train_one(args, entry, "quick")
        write_report(args)


def run_full(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    state_path = out_root / STATE_NAME
    entries = select_full_entries(args, int(args.top_full))
    if not entries:
        return
    completed = {str(r.get("spec", {}).get("id", "")) for r in load_rows(out_root / SUMMARY_NAME) if r.get("mode") == "full"}
    for i, entry in enumerate(entries, start=1):
        spec_id = str(entry["spec"]["id"])
        if spec_id in completed:
            continue
        update_state(state_path, status="full_training", stage="full_train", current=spec_id, index=i, total=len(entries), updated_at=now_iso())
        train_one(args, entry, "full")
        write_report(args)


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / STATE_NAME
    update_state(state_path, status="running", stage=args.stage, updated_at=now_iso())
    if args.stage in {"quick_train", "all"}:
        run_quick(args)
    if args.stage in {"full_train", "all"}:
        run_full(args)
    if args.stage in {"report", "all"}:
        write_report(args)
    update_state(state_path, status="complete", stage=args.stage, updated_at=now_iso())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BUT 10s fatal-OR morphology logic grid.")
    parser.add_argument("--stage", choices=("quick_train", "full_train", "report", "all"), default="all")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--protocol_out_root", default=str(PROTOCOL_OUT_ROOT))
    parser.add_argument("--protocol_report_root", default=str(PROTOCOL_REPORT_ROOT))
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_runs", type=int, default=18)
    parser.add_argument("--top_full", type=int, default=4)
    parser.add_argument("--quick_epochs_stage1", type=int, default=4)
    parser.add_argument("--quick_epochs_stage2", type=int, default=4)
    parser.add_argument("--full_epochs_stage1", type=int, default=10)
    parser.add_argument("--full_epochs_stage2", type=int, default=8)
    parser.add_argument("--batch_size_stage1", type=int, default=24)
    parser.add_argument("--batch_size_stage2", type=int, default=32)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--lambda_den_stage2", type=float, default=0.5)
    parser.add_argument("--lambda_cls", type=float, default=18.0)
    parser.add_argument("--min_ptb_acc", type=float, default=0.965)
    parser.add_argument("--min_ptb_bad", type=float, default=0.985)
    parser.add_argument("--audit_max_per_class", type=int, default=400)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        run(args)
        print(json.dumps({"status": "complete", "out_root": args.out_root}, ensure_ascii=False), flush=True)
    except Exception as exc:
        update_state(Path(args.out_root) / STATE_NAME, status="failed", error=str(exc), updated_at=now_iso())
        raise


if __name__ == "__main__":
    main()
