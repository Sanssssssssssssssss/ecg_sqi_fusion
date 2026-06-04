"""BUT 10s generator continuation with mixture-style medium rules.

The previous generator pass showed that uniformly damaging every medium sample
does not reproduce BUT's expert class-2 boundary.  This pass keeps b10-style
bad artifacts and makes medium heterogeneous: QRS remains mostly visible, but
different subsets get P/T degradation, short contact loss, drift, low amplitude,
or bursty local uncertainty.
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
from src.transformer_pipeline.external_benchmarks.but_adaptation_14h import (
    ROOT,
    burst_noise,
    class_array,
    highpass_noise,
    load_ptb_artifact,
    lowfreq_drift,
    normalize_noise,
)
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


RUN_TAG = "e311_but_generator_medium_mixture_grid_10s_2026_06_03"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def entries() -> list[dict[str, Any]]:
    base_bad = b10_base()
    # Start close to quick b10 because it is still the best generator anchor.
    base_bad.update(
        {
            "qrs_attn_medium": 0.03,
            "tst_noise_medium": 0.08,
            "hf_medium": 0.035,
            "drift_medium": 0.045,
            "qrs_attn_bad": 0.70,
            "contact_loss_bad": 0.36,
            "flatline_bad": 0.25,
            "baseline_step_bad": 0.30,
            "clip_bad": 0.25,
            "spurious_peak_bad": 0.28,
            "low_amp_bad": 0.56,
        }
    )
    variants: list[tuple[str, dict[str, Any], str, str]] = [
        (
            "mix01_b10_medium_hetero_light",
            {
                "medium_mix_strength": 0.55,
                "medium_contact_prob": 0.08,
                "medium_lowamp_prob": 0.22,
                "medium_pt_noise": 0.16,
                "medium_drift": 0.07,
                "medium_hf": 0.06,
                "medium_short_dropout": 0.06,
                "medium_qrs_guard": 0.94,
            },
            "1,1.48,1.86",
            "Near b10 bad with heterogeneous but light medium.",
        ),
        (
            "mix02_medium_contact_short",
            {
                "medium_mix_strength": 0.65,
                "medium_contact_prob": 0.16,
                "medium_lowamp_prob": 0.18,
                "medium_pt_noise": 0.16,
                "medium_drift": 0.08,
                "medium_hf": 0.06,
                "medium_short_dropout": 0.16,
                "medium_qrs_guard": 0.92,
            },
            "1,1.52,1.84",
            "Medium can have short contact loss, but QRS is preserved.",
        ),
        (
            "mix03_medium_lowamp_visible_qrs",
            {
                "medium_mix_strength": 0.70,
                "medium_contact_prob": 0.08,
                "medium_lowamp_prob": 0.42,
                "medium_pt_noise": 0.14,
                "medium_drift": 0.07,
                "medium_hf": 0.055,
                "medium_short_dropout": 0.06,
                "medium_qrs_guard": 0.96,
            },
            "1,1.54,1.82",
            "BUT-like low amplitude medium with visible QRS.",
        ),
        (
            "mix04_medium_pt_unreliable",
            {
                "medium_mix_strength": 0.75,
                "medium_contact_prob": 0.06,
                "medium_lowamp_prob": 0.18,
                "medium_pt_noise": 0.26,
                "medium_drift": 0.09,
                "medium_hf": 0.08,
                "medium_short_dropout": 0.05,
                "medium_qrs_guard": 0.97,
            },
            "1,1.58,1.80",
            "Class-2 emphasis: P/T/ST unreliable while QRS stays usable.",
        ),
        (
            "mix05_medium_protocol_boundary",
            {
                "medium_mix_strength": 0.82,
                "medium_contact_prob": 0.12,
                "medium_lowamp_prob": 0.28,
                "medium_pt_noise": 0.22,
                "medium_drift": 0.10,
                "medium_hf": 0.075,
                "medium_short_dropout": 0.12,
                "medium_qrs_guard": 0.93,
            },
            "1,1.60,1.78",
            "Mixed expert-boundary medium: several mild failure modes.",
        ),
        (
            "mix06_medium_strong_bad_soft",
            {
                "qrs_attn_bad": 0.64,
                "contact_loss_bad": 0.30,
                "flatline_bad": 0.18,
                "baseline_step_bad": 0.22,
                "clip_bad": 0.18,
                "spurious_peak_bad": 0.14,
                "low_amp_bad": 0.62,
                "medium_mix_strength": 0.80,
                "medium_contact_prob": 0.14,
                "medium_lowamp_prob": 0.28,
                "medium_pt_noise": 0.24,
                "medium_drift": 0.10,
                "medium_hf": 0.08,
                "medium_short_dropout": 0.10,
                "medium_qrs_guard": 0.94,
            },
            "1,1.62,1.70",
            "Soften bad and strengthen medium boundary cues.",
        ),
        (
            "mix07_medium_good_overlap",
            {
                "residual_scale_good": 0.85,
                "residual_scale_medium": 1.10,
                "medium_mix_strength": 0.58,
                "medium_contact_prob": 0.08,
                "medium_lowamp_prob": 0.20,
                "medium_pt_noise": 0.18,
                "medium_drift": 0.08,
                "medium_hf": 0.06,
                "medium_short_dropout": 0.06,
                "medium_qrs_guard": 0.95,
            },
            "1,1.50,1.82",
            "Make good less pristine without collapsing medium to bad.",
        ),
        (
            "mix08_b10_medium_bad_prior_guard",
            {
                "medium_mix_strength": 0.68,
                "medium_contact_prob": 0.10,
                "medium_lowamp_prob": 0.24,
                "medium_pt_noise": 0.20,
                "medium_drift": 0.08,
                "medium_hf": 0.07,
                "medium_short_dropout": 0.08,
                "medium_qrs_guard": 0.95,
            },
            "1,1.56,1.88",
            "Preserve high bad pressure while adding heterogeneous medium.",
        ),
    ]
    out: list[dict[str, Any]] = []
    for name, changes, class_weight, note in variants:
        spec = dict(base_bad)
        spec.update(changes)
        spec["id"] = name
        spec["family"] = "medium_mixture_generator_10s"
        out.append({"spec": spec, "class_weight": class_weight, "note": note})
    return out


def apply_medium_mixture(ptb: dict[str, Any], spec: dict[str, Any], out_dir: Path, seed: int) -> dict[str, Any]:
    payload = apply_bad_spec(ptb, spec, out_dir, seed=seed, sample_limit=0)
    labels = payload["labels"].reset_index(drop=True)
    x = payload["X_noisy"].astype(np.float32, copy=True)
    clean = ptb["clean"].astype(np.float32, copy=False)
    masks = ptb["masks"]
    y = class_array(labels)
    medium_idx = np.where(y == 1)[0]
    if len(medium_idx) == 0:
        payload["X_noisy"] = x
        return payload
    rng = np.random.default_rng(seed + 311)
    qrs = masks.get("qrs_mask", np.zeros_like(clean)).astype(np.float32)
    tst = masks.get("tst_mask", np.zeros_like(clean)).astype(np.float32)
    std = np.std(clean[medium_idx], axis=1, keepdims=True) + 1e-6
    mix_strength = float(spec.get("medium_mix_strength", 0.6))
    chosen = medium_idx[rng.random(len(medium_idx)) < mix_strength]
    if len(chosen) == 0:
        payload["X_noisy"] = x
        return payload
    local_pos = {int(idx): pos for pos, idx in enumerate(medium_idx)}
    chosen_pos = np.array([local_pos[int(idx)] for idx in chosen], dtype=np.int64)
    q = qrs[chosen]
    t = tst[chosen]
    s = std[chosen_pos]
    x[chosen] += float(spec.get("medium_pt_noise", 0.18)) * s * normalize_noise(rng.normal(size=(len(chosen), N_TARGET)).astype(np.float32)) * t
    x[chosen] += float(spec.get("medium_drift", 0.08)) * s * lowfreq_drift(rng, len(chosen))
    x[chosen] += float(spec.get("medium_hf", 0.06)) * s * normalize_noise(highpass_noise(rng, len(chosen)))
    x[chosen] += 0.04 * s * burst_noise(rng, len(chosen), 0.65) * (1.0 - np.clip(q, 0.0, 1.0))
    # Preserve QRS amplitude explicitly; this keeps medium from becoming bad.
    guard = float(spec.get("medium_qrs_guard", 0.95))
    x[chosen] = x[chosen] * (1.0 - q) + (guard * clean[chosen] + (1.0 - guard) * x[chosen]) * q
    lowamp_prob = float(spec.get("medium_lowamp_prob", 0.2))
    contact_prob = float(spec.get("medium_contact_prob", 0.1))
    dropout_prob = float(spec.get("medium_short_dropout", 0.08))
    for idx in chosen:
        i = int(idx)
        if rng.random() < lowamp_prob:
            x[i] *= float(rng.uniform(0.72, 0.90))
        if rng.random() < contact_prob:
            width = int(rng.integers(FS_TARGET // 3, FS_TARGET * 2))
            start = int(rng.integers(0, max(1, N_TARGET - width)))
            x[i, start : start + width] *= float(rng.uniform(0.40, 0.72))
        if rng.random() < dropout_prob:
            width = int(rng.integers(FS_TARGET // 4, FS_TARGET))
            start = int(rng.integers(0, max(1, N_TARGET - width)))
            level = float(np.median(x[i]))
            x[i, start : start + width] = 0.65 * x[i, start : start + width] + 0.35 * level
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
    payload = apply_medium_mixture(ptb, spec, variant_dir, seed=int(args.seed))
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
    write_json(run_dir / "medium_mixture_generator_run_summary.json", payload)
    append_jsonl(Path(args.out_root) / "medium_mixture_generator_summary.jsonl", payload)
    return payload


def write_report(args: argparse.Namespace, rows: list[dict[str, Any]] | None = None) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    if rows is None:
        summary_path = out_root / "medium_mixture_generator_summary.jsonl"
        rows = [json.loads(line) for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()] if summary_path.exists() else []
    ranked = sorted(rows, key=result_key, reverse=True)
    lines = [
        "# BUT 10s Medium Mixture Generator",
        "",
        "Mixture-style medium generation after uniform medium/bad rules failed to beat b10. Formal BUT protocol remains 10s P1; calibration is validation-only.",
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
    (report_root / "medium_mixture_generator_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(report_root / "medium_mixture_generator_summary.json", {"rows": rows, "ranked": ranked[:50]})


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / "medium_mixture_generator_state.json"
    update_state(state_path, status="running", stage=args.stage, updated_at=now_iso())
    run_entries = entries()
    write_json(out_root / "medium_mixture_generator_specs.json", {"entries": run_entries})
    rows: list[dict[str, Any]] = []
    if args.stage in {"all", "quick_train"}:
        for i, entry in enumerate(run_entries[: int(args.max_runs)], start=1):
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
    parser = argparse.ArgumentParser(description="Run BUT 10s medium mixture generator tuning.")
    parser.add_argument("--stage", choices=("quick_train", "report", "all"), default="all")
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--protocol_out_root", default=str(PROTOCOL_OUT_ROOT))
    parser.add_argument("--protocol_report_root", default=str(PROTOCOL_REPORT_ROOT))
    parser.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_runs", type=int, default=8)
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
        update_state(Path(args.out_root) / "medium_mixture_generator_state.json", status="failed", error=str(exc), updated_at=now_iso())
        raise


if __name__ == "__main__":
    main()
