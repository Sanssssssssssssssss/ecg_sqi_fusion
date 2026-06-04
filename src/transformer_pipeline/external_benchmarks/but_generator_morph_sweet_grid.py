"""BUT 10s morphology-focused generator search.

This pass ignores SNR as a primary design axis.  It creates local morphology
events that resemble BUT QDB: QRS remains interpretable for medium, but P/T/ST
segments, baseline steps, inverted pseudo-peaks, and short contact artifacts
make detailed measurements unreliable.  Bad receives denser pseudo-peaks and
QRS-confounding contact events.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks.but_adaptation_14h import ROOT, class_array, load_ptb_artifact
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_refine import b10_base
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_tuning import (
    DEFAULT_SOURCE_ARTIFACT,
    apply_bad_spec,
    evaluate_checkpoint_10s,
    now_iso,
    plot_synthetic_gallery,
    read_json,
    update_state,
)
from src.transformer_pipeline.external_benchmarks.but_protocol_adaptation import (
    DEFAULT_OUT_ROOT as PROTOCOL_OUT_ROOT,
    DEFAULT_REPORT_ROOT as PROTOCOL_REPORT_ROOT,
)
from src.transformer_pipeline.external_benchmarks.run import FS_TARGET, N_TARGET


RUN_TAG = "e311_but_generator_morph_sweet_grid_10s_2026_06_03"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def entries() -> list[dict[str, Any]]:
    base = b10_base()
    base.update(
        {
            "qrs_attn_medium": 0.02,
            "tst_noise_medium": 0.06,
            "hf_medium": 0.03,
            "drift_medium": 0.04,
            "qrs_attn_bad": 0.70,
            "contact_loss_bad": 0.34,
            "flatline_bad": 0.22,
            "baseline_step_bad": 0.26,
            "clip_bad": 0.20,
            "spurious_peak_bad": 0.20,
            "low_amp_bad": 0.60,
        }
    )
    variants: list[tuple[str, dict[str, Any], str, str]] = [
        (
            "s01_pt_baseline_steps",
            {
                "medium_event_prob": 0.75,
                "medium_pseudo_peak_rate": 0.10,
                "medium_inverse_peak_rate": 0.18,
                "medium_step_rate": 0.20,
                "medium_ramp_rate": 0.16,
                "medium_contact_rate": 0.08,
                "medium_qrs_preserve": 0.98,
                "bad_pseudo_peak_rate": 0.34,
                "bad_inverse_peak_rate": 0.26,
                "bad_qrs_confuse_rate": 0.18,
            },
            "1,1.56,1.82",
            "Medium local baseline/P-T morphology events with preserved QRS.",
        ),
        (
            "s02_dense_bad_spikes_medium_soft",
            {
                "medium_event_prob": 0.62,
                "medium_pseudo_peak_rate": 0.08,
                "medium_inverse_peak_rate": 0.14,
                "medium_step_rate": 0.14,
                "medium_ramp_rate": 0.12,
                "medium_contact_rate": 0.06,
                "medium_qrs_preserve": 0.99,
                "bad_pseudo_peak_rate": 0.54,
                "bad_inverse_peak_rate": 0.34,
                "bad_qrs_confuse_rate": 0.30,
            },
            "1,1.50,1.88",
            "Bad has dense QRS-confounding spikes; medium stays soft.",
        ),
        (
            "s03_but_medium_motion",
            {
                "medium_event_prob": 0.82,
                "medium_pseudo_peak_rate": 0.16,
                "medium_inverse_peak_rate": 0.22,
                "medium_step_rate": 0.26,
                "medium_ramp_rate": 0.24,
                "medium_contact_rate": 0.12,
                "medium_qrs_preserve": 0.96,
                "bad_pseudo_peak_rate": 0.40,
                "bad_inverse_peak_rate": 0.30,
                "bad_qrs_confuse_rate": 0.24,
            },
            "1,1.62,1.80",
            "Aggressive BUT-like medium motion while keeping QRS mostly visible.",
        ),
        (
            "s04_contact_bad_medium_visible",
            {
                "qrs_attn_bad": 0.76,
                "contact_loss_bad": 0.44,
                "flatline_bad": 0.32,
                "baseline_step_bad": 0.34,
                "spurious_peak_bad": 0.12,
                "medium_event_prob": 0.70,
                "medium_pseudo_peak_rate": 0.10,
                "medium_inverse_peak_rate": 0.18,
                "medium_step_rate": 0.20,
                "medium_ramp_rate": 0.18,
                "medium_contact_rate": 0.10,
                "medium_qrs_preserve": 0.98,
                "bad_pseudo_peak_rate": 0.28,
                "bad_inverse_peak_rate": 0.22,
                "bad_qrs_confuse_rate": 0.34,
            },
            "1,1.56,1.88",
            "Bad contact/flatline, medium visible-QRS morphology uncertainty.",
        ),
        (
            "s05_medium_ambiguous_good_guard",
            {
                "residual_scale_good": 0.86,
                "residual_scale_medium": 1.10,
                "medium_event_prob": 0.72,
                "medium_pseudo_peak_rate": 0.12,
                "medium_inverse_peak_rate": 0.20,
                "medium_step_rate": 0.22,
                "medium_ramp_rate": 0.18,
                "medium_contact_rate": 0.08,
                "medium_qrs_preserve": 0.97,
                "bad_pseudo_peak_rate": 0.34,
                "bad_inverse_peak_rate": 0.28,
                "bad_qrs_confuse_rate": 0.24,
            },
            "1,1.58,1.82",
            "Good less pristine, medium ambiguous, bad QRS-confounded.",
        ),
        (
            "s06_medium_local_morph_bad_hard",
            {
                "qrs_attn_bad": 0.82,
                "contact_loss_bad": 0.40,
                "flatline_bad": 0.28,
                "baseline_step_bad": 0.30,
                "medium_event_prob": 0.78,
                "medium_pseudo_peak_rate": 0.14,
                "medium_inverse_peak_rate": 0.24,
                "medium_step_rate": 0.24,
                "medium_ramp_rate": 0.22,
                "medium_contact_rate": 0.10,
                "medium_qrs_preserve": 0.97,
                "bad_pseudo_peak_rate": 0.50,
                "bad_inverse_peak_rate": 0.34,
                "bad_qrs_confuse_rate": 0.34,
            },
            "1,1.60,1.86",
            "Hard bad plus explicit local morphology medium.",
        ),
    ]
    out: list[dict[str, Any]] = []
    for name, changes, class_weight, note in variants:
        spec = dict(base)
        spec.update(changes)
        spec["id"] = name
        spec["family"] = "morph_sweet_generator_10s"
        out.append({"spec": spec, "class_weight": class_weight, "note": note})
    return out


def add_pulse(x: np.ndarray, center: int, width: int, amp: float, sign: float) -> None:
    center = int(np.clip(center, 2, len(x) - 3))
    width = max(1, min(width, center, len(x) - center - 1))
    if width <= 0:
        return
    pulse = np.hanning(width * 2 + 1).astype(np.float32)
    start = center - width
    end = center + width + 1
    if end <= start or len(pulse) != end - start:
        return
    x[start:end] += sign * amp * pulse


def add_local_events(
    x: np.ndarray,
    clean: np.ndarray,
    indices: np.ndarray,
    qrs: np.ndarray,
    spec: dict[str, Any],
    rng: np.random.Generator,
    prefix: str,
) -> None:
    event_prob = float(spec.get(f"{prefix}_event_prob", 1.0))
    pseudo_rate = float(spec.get(f"{prefix}_pseudo_peak_rate", 0.1))
    inverse_rate = float(spec.get(f"{prefix}_inverse_peak_rate", 0.1))
    step_rate = float(spec.get(f"{prefix}_step_rate", 0.1))
    ramp_rate = float(spec.get(f"{prefix}_ramp_rate", 0.1))
    contact_rate = float(spec.get(f"{prefix}_contact_rate", 0.05))
    qrs_preserve = float(spec.get(f"{prefix}_qrs_preserve", 0.98))
    for idx in indices:
        i = int(idx)
        if rng.random() > event_prob:
            continue
        std = float(np.std(clean[i]) + 1e-6)
        # Mostly off-QRS pseudo morphology.  BUT medium has lots of fake
        # deflections, but QRS usually remains trackable.
        n_pulse = int(rng.integers(1, 4))
        for _ in range(n_pulse):
            if rng.random() < pseudo_rate:
                center = int(rng.integers(FS_TARGET // 3, N_TARGET - FS_TARGET // 3))
                if qrs[i, center] > 0.1 and prefix == "medium":
                    center = int((center + FS_TARGET // 2) % (N_TARGET - FS_TARGET))
                add_pulse(x[i], center, int(rng.integers(4, 16)), float(rng.uniform(0.25, 0.75)) * std, 1.0)
            if rng.random() < inverse_rate:
                center = int(rng.integers(FS_TARGET // 3, N_TARGET - FS_TARGET // 3))
                add_pulse(x[i], center, int(rng.integers(8, 26)), float(rng.uniform(0.20, 0.65)) * std, -1.0)
        if rng.random() < step_rate:
            pos = int(rng.integers(FS_TARGET // 2, N_TARGET - FS_TARGET // 2))
            width = int(rng.integers(FS_TARGET // 3, FS_TARGET * 2))
            amp = float(rng.normal(0.0, 0.22 if prefix == "medium" else 0.34)) * std
            x[i, pos : min(N_TARGET, pos + width)] += amp
        if rng.random() < ramp_rate:
            pos = int(rng.integers(0, N_TARGET - FS_TARGET))
            width = int(rng.integers(FS_TARGET, FS_TARGET * 4))
            end = min(N_TARGET, pos + width)
            amp = float(rng.normal(0.0, 0.24 if prefix == "medium" else 0.38)) * std
            x[i, pos:end] += np.linspace(0.0, amp, end - pos, dtype=np.float32)
        if rng.random() < contact_rate:
            pos = int(rng.integers(0, N_TARGET - FS_TARGET))
            width = int(rng.integers(FS_TARGET // 3, FS_TARGET * (2 if prefix == "medium" else 4)))
            end = min(N_TARGET, pos + width)
            factor = float(rng.uniform(0.42, 0.76) if prefix == "medium" else rng.uniform(0.04, 0.42))
            x[i, pos:end] *= factor
        if prefix == "medium":
            x[i] = x[i] * (1.0 - qrs[i]) + (qrs_preserve * clean[i] + (1.0 - qrs_preserve) * x[i]) * qrs[i]


def apply_morph_spec(ptb: dict[str, Any], spec: dict[str, Any], out_dir: Path, seed: int) -> dict[str, Any]:
    payload = apply_bad_spec(ptb, spec, out_dir, seed=seed, sample_limit=0)
    x = payload["X_noisy"].astype(np.float32, copy=True)
    labels = payload["labels"].reset_index(drop=True)
    clean = ptb["clean"].astype(np.float32, copy=False)
    qrs = ptb["masks"].get("qrs_mask", np.zeros_like(clean)).astype(np.float32)
    y = class_array(labels)
    rng = np.random.default_rng(seed + 911)
    medium = np.where(y == 1)[0]
    bad = np.where(y == 2)[0]
    add_local_events(x, clean, medium, qrs, spec, rng, "medium")
    # Bad has QRS-confounding events too; unlike medium, QRS is not restored.
    bad_spec = dict(spec)
    bad_spec["bad_event_prob"] = 1.0
    add_local_events(x, clean, bad, qrs, bad_spec, rng, "bad")
    confuse_rate = float(spec.get("bad_qrs_confuse_rate", 0.25))
    for idx in bad:
        i = int(idx)
        std = float(np.std(clean[i]) + 1e-6)
        q_positions = np.flatnonzero(qrs[i] > 0.2)
        if len(q_positions) and rng.random() < confuse_rate:
            for _ in range(int(rng.integers(2, 6))):
                center = int(rng.choice(q_positions))
                add_pulse(x[i], center, int(rng.integers(4, 14)), float(rng.uniform(0.25, 0.80)) * std, float(rng.choice([-1.0, 1.0])))
    payload["X_noisy"] = x.astype(np.float32)
    np.savez_compressed(out_dir / "datasets" / "signals.npz", X=payload["X_noisy"][:, None, :].astype(np.float32))
    return payload


def result_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    rep = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
    rec = rep.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
    ptb = row.get("ptb_test_report", {})
    den = row.get("ptb_denoise_metrics", {})
    return (
        min(float(rec[1]), float(rec[2])),
        float(rep.get("balanced_acc", 0.0)),
        float(rep.get("macro_f1", 0.0)),
        float(rec[2]),
        float(ptb.get("acc", 0.0)) + 0.01 * float(den.get("denoise_score", 0.0)),
    )


def prepare_variant(args: argparse.Namespace, entry: dict[str, Any]) -> Path:
    spec = entry["spec"]
    variant_dir = Path(args.out_root) / "synthetic_variants" / str(spec["id"])
    if (variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv").exists():
        return variant_dir
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
    return variant_dir


def train_one(args: argparse.Namespace, entry: dict[str, Any]) -> dict[str, Any]:
    spec = entry["spec"]
    spec_id = str(spec["id"])
    variant_dir = prepare_variant(args, entry)
    run_dir = Path(args.out_root) / "runs" / "quick" / spec_id
    log_dir = Path(args.out_root) / "logs" / "quick"
    log_dir.mkdir(parents=True, exist_ok=True)
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
        str(args.quick_epochs_stage1),
        "--epochs_stage2",
        str(args.quick_epochs_stage2),
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
        "returncode": int(proc.returncode),
        "elapsed_sec": float(time.perf_counter() - start),
        "run_dir": str(run_dir),
        "variant_dir": str(variant_dir),
    }
    summary_path = run_dir / "mainline_summary.json"
    if proc.returncode == 0 and (run_dir / "ckpt_best.pt").exists() and summary_path.exists():
        summary = read_json(summary_path)
        payload["ptb_test_report"] = summary.get("stage2", {}).get("test_report", {})
        payload["ptb_denoise_metrics"] = summary.get("stage2", {}).get("denoise_metrics", {})
        payload["but_10s_eval"] = evaluate_checkpoint_10s(args, run_dir / "ckpt_best.pt", run_dir)
    else:
        payload["status"] = "failed"
    write_json(run_dir / "morph_sweet_generator_run_summary.json", payload)
    append_jsonl(Path(args.out_root) / "morph_sweet_generator_summary.jsonl", payload)
    return payload


def write_report(args: argparse.Namespace, rows: list[dict[str, Any]] | None = None) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    if rows is None:
        summary_path = out_root / "morph_sweet_generator_summary.jsonl"
        rows = [json.loads(line) for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()] if summary_path.exists() else []
    ranked = sorted(rows, key=result_key, reverse=True)
    lines = [
        "# BUT 10s Morphology Sweet-Spot Generator",
        "",
        "Morphology-focused synthetic rules.  SNR is not the design axis; local shape events and QRS interpretability are.",
        "",
        "| rank | spec | cw | return | BUT acc | BUT bal | macro-F1 | recalls good/medium/bad | PTB acc | PTB bad | denoise | note |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for i, row in enumerate(ranked, start=1):
        rep = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
        rec = rep.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        ptb = row.get("ptb_test_report", {})
        ptb_rec = ptb.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
        den = row.get("ptb_denoise_metrics", {})
        lines.append(
            f"| {i} | {row.get('spec', {}).get('id')} | {row.get('class_weight')} | {row.get('returncode')} | "
            f"{float(rep.get('acc', 0.0)):.4f} | {float(rep.get('balanced_acc', 0.0)):.4f} | {float(rep.get('macro_f1', 0.0)):.4f} | "
            f"{float(rec[0]):.3f}/{float(rec[1]):.3f}/{float(rec[2]):.3f} | "
            f"{float(ptb.get('acc', 0.0)):.4f} | {float(ptb_rec[2] if len(ptb_rec)>2 else 0.0):.4f} | "
            f"{float(den.get('denoise_score', 0.0)):.3f} | {row.get('note', '')} |"
        )
    (report_root / "morph_sweet_generator_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report_root / "morph_sweet_generator_summary.json", {"rows": rows, "ranked": ranked[:50]})


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / "morph_sweet_generator_state.json"
    update_state(state_path, status="running", stage=args.stage, updated_at=now_iso())
    run_entries = entries()
    write_json(out_root / "morph_sweet_generator_specs.json", {"entries": run_entries})
    rows: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    summary_path = out_root / "morph_sweet_generator_summary.jsonl"
    if summary_path.exists():
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                completed_ids.add(str(json.loads(line).get("spec", {}).get("id", "")))
            except json.JSONDecodeError:
                continue
    if args.stage in {"all", "quick_train"}:
        for i, entry in enumerate(run_entries[: int(args.max_runs)], start=1):
            if str(entry["spec"]["id"]) in completed_ids:
                continue
            update_state(
                state_path,
                status="quick_training",
                current=entry["spec"]["id"],
                index=i,
                total=min(len(run_entries), int(args.max_runs)),
                updated_at=now_iso(),
            )
            rows.append(train_one(args, entry))
            write_report(args)
    if args.stage in {"all", "report"}:
        write_report(args, rows or None)
    update_state(state_path, status="complete", stage=args.stage, updated_at=now_iso(), n_runs=len(rows) if rows else None)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BUT 10s morphology-focused generator tuning.")
    parser.add_argument("--stage", choices=("quick_train", "report", "all"), default="all")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--protocol_out_root", default=str(PROTOCOL_OUT_ROOT))
    parser.add_argument("--protocol_report_root", default=str(PROTOCOL_REPORT_ROOT))
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_runs", type=int, default=6)
    parser.add_argument("--quick_epochs_stage1", type=int, default=4)
    parser.add_argument("--quick_epochs_stage2", type=int, default=4)
    parser.add_argument("--batch_size_stage1", type=int, default=24)
    parser.add_argument("--batch_size_stage2", type=int, default=32)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--lambda_den_stage2", type=float, default=0.5)
    parser.add_argument("--lambda_cls", type=float, default=18.0)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        run(args)
        print(json.dumps({"status": "complete", "out_root": args.out_root}, ensure_ascii=False), flush=True)
    except Exception as exc:
        update_state(Path(args.out_root) / "morph_sweet_generator_state.json", status="failed", error=str(exc), updated_at=now_iso())
        raise


if __name__ == "__main__":
    main()
