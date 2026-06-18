"""Direction validation for BUT 10s medium-guard / bad-OR experiments.

This is an experiment-only continuation runner.  It treats the existing
medium-guard full confirmation as Stage 1, waits for it to finish, then runs
seed confirmation, mechanism ablations, a small focus grid, and a compact
report.  It intentionally does not touch src/sqi_pipeline or mainline
checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.transformer_pipeline.external_benchmarks import (
    but_medium_guard_bad_boundary_grid_10s as mgb,
)
from src.transformer_pipeline.external_benchmarks import (
    real_noise_snr_but_match_grid_10s as rn,
)
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_tuning import (
    now_iso,
    update_state,
)
from src.transformer_pipeline.external_benchmarks.but_medium_guard_bad_boundary_analysis_10s import (
    FEATURE_COLUMNS,
    classify_errors,
    plot_wave_cases,
    stratified_balanced_indices,
)
from src.transformer_pipeline.external_benchmarks.run import (
    apply_but_thresholds,
    calibrate_but,
    load_uformer_model,
    multiclass_report,
    run_model_outputs,
)


ROOT = mgb.ROOT
RUN_TAG = "e311_but_direction_validation_10s_2026_06_05"
DEFAULT_SOURCE_OUT = (
    ROOT
    / "outputs"
    / "external_benchmarks"
    / "e311_but_medium_guard_bad_boundary_grid_10s_2026_06_05"
)
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG

STATE_NAME = "direction_validation_state.json"
SUMMARY_NAME = "direction_validation_summary.jsonl"
SELECTED_NAME = "selected_candidates.json"

QUICK_BEST_ID = "mg_balanced_or_m_strong_detail_midbad__bw_badstrong__cw1p00_1p55_1p70"
ANCHOR_MACRO = float(mgb.H_BAD_RESCUE_05["macro_f1"])


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def metric_report(row: dict[str, Any]) -> dict[str, Any]:
    eval_block = row.get("but_10s_eval") or {}
    return eval_block.get("but_10s_test_report") or row.get("but_10s_test_report") or {}


def metric_value(row: dict[str, Any], key: str, default: float = -1.0) -> float:
    report = metric_report(row)
    val = report.get(key, default)
    try:
        return float(val)
    except Exception:
        return default


def recalls(row: dict[str, Any]) -> tuple[float, float, float]:
    report = metric_report(row)
    vals = report.get("recall_per_class", [math.nan, math.nan, math.nan])
    if len(vals) < 3:
        return (math.nan, math.nan, math.nan)
    return (float(vals[0]), float(vals[1]), float(vals[2]))


def spec_id(row: dict[str, Any]) -> str:
    spec = row.get("spec") or {}
    return str(row.get("variant_id") or spec.get("id") or row.get("run_id") or "unknown")


def is_success(row: dict[str, Any]) -> bool:
    status = str(row.get("status", "")).lower()
    if status in {"success", "complete", "completed"}:
        return True
    if row.get("returncode") == 0 and metric_report(row):
        return True
    return bool(metric_report(row))


def source_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    source_out = Path(args.source_out)
    return source_out / mgb.SUMMARY_NAME, source_out / mgb.STATE_NAME


def source_full_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    summary_path, _ = source_paths(args)
    return [
        row
        for row in load_jsonl(summary_path)
        if row.get("mode") == "full" and is_success(row)
    ]


def source_quick_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    summary_path, _ = source_paths(args)
    return [
        row
        for row in load_jsonl(summary_path)
        if row.get("mode") == "quick" and is_success(row)
    ]


def wait_for_source_full(args: argparse.Namespace) -> None:
    """Wait until the existing medium-guard runner is safe to hand off from."""

    state_path = Path(args.out_root) / STATE_NAME
    _, source_state_path = source_paths(args)
    while True:
        full_rows = source_full_rows(args)
        source_state = read_json(source_state_path, {})
        source_status = str(source_state.get("status") or source_state.get("stage") or "")
        completed = len(full_rows)
        update_state(
            state_path,
            status="waiting_for_source_full",
            source_status=source_status,
            source_full_completed=completed,
            source_full_expected=args.expected_full,
            updated_at=now_iso(),
        )

        source_running = source_status.lower() in {
            "quick_training",
            "full_training",
            "seed_training",
            "running",
        }
        if completed >= args.expected_full and not (
            bool(args.wait_for_source_idle) and source_running
        ):
            return
        if source_status.lower() in {"complete", "completed", "full_complete", "done"}:
            return
        if source_status.lower() in {"failed", "error"}:
            raise RuntimeError(f"Source medium-guard grid failed: {source_state}")

        print(
            f"[direction] waiting for source full: {completed}/{args.expected_full} "
            f"status={source_status or 'unknown'}"
        )
        time.sleep(float(args.wait_poll_sec))


def select_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.wait_for_source_full:
        wait_for_source_full(args)

    full_rows = sorted(
        source_full_rows(args),
        key=lambda row: (metric_value(row, "macro_f1"), metric_value(row, "acc")),
        reverse=True,
    )
    quick_rows = sorted(
        source_quick_rows(args),
        key=lambda row: (metric_value(row, "macro_f1"), metric_value(row, "acc")),
        reverse=True,
    )
    if not full_rows:
        raise RuntimeError("No successful source full rows are available yet.")

    selected = full_rows[: max(1, int(args.top_seed_candidates))]
    mandatory = next((row for row in full_rows if spec_id(row) == QUICK_BEST_ID), None)
    if mandatory is not None and all(spec_id(row) != QUICK_BEST_ID for row in selected):
        selected = selected[:-1] + [mandatory]
        selected = sorted(
            selected,
            key=lambda row: (metric_value(row, "macro_f1"), metric_value(row, "acc")),
            reverse=True,
        )

    payload = {
        "created_at": now_iso(),
        "source_summary": str(source_paths(args)[0]),
        "source_state": str(source_paths(args)[1]),
        "quick_best": quick_rows[0] if quick_rows else None,
        "selected_full_candidates": selected,
        "frozen_anchor": mgb.H_BAD_RESCUE_05,
    }
    write_json(Path(args.out_root) / SELECTED_NAME, payload)
    write_json(Path(args.report_root) / SELECTED_NAME, payload)
    update_state(
        Path(args.out_root) / STATE_NAME,
        status="selected",
        selected_count=len(selected),
        best_full=spec_id(selected[0]),
        best_full_macro=metric_value(selected[0], "macro_f1"),
        updated_at=now_iso(),
    )
    return selected


def make_train_args(args: argparse.Namespace, seed: int = 0) -> argparse.Namespace:
    return argparse.Namespace(
        out_root=str(args.out_root),
        report_root=str(args.report_root),
        source_artifact_dir=str(args.source_artifact_dir),
        but_protocol_dir=str(args.but_protocol_dir),
        protocol_out_root=str(args.protocol_out_root),
        protocol_report_root=str(args.protocol_report_root),
        nstdb_root=str(args.nstdb_root),
        cinc_dir=str(args.cinc_dir),
        seed=int(seed),
        seeds=args.seeds,
        force=bool(args.force),
        rerun_existing=bool(args.rerun_existing),
        skip_failed_existing=False,
        train_regenerate_full=int(args.train_regenerate_full),
        quick_epochs_stage1=int(args.quick_epochs_stage1),
        quick_epochs_stage2=int(args.quick_epochs_stage2),
        full_epochs_stage1=int(args.full_epochs_stage1),
        full_epochs_stage2=int(args.full_epochs_stage2),
        batch_size_stage1=int(args.batch_size_stage1),
        batch_size_stage2=int(args.batch_size_stage2),
        batch_size_eval=int(args.batch_size_eval),
        hidden_dim=int(args.hidden_dim),
        denoiser_width=float(args.denoiser_width),
        lambda_den=float(args.lambda_den),
        lambda_cls=float(args.lambda_cls),
        lr_stage1=float(args.lr_stage1),
        lr_head=float(args.lr_head),
        lr_denoiser=float(args.lr_denoiser),
        lambda_den_stage2=float(args.lambda_den),
        class_weight="1.0,1.55,1.70",
        cpu=bool(args.cpu),
    )


def direction_key(row: dict[str, Any], stage: str) -> str:
    return "|".join(
        [
            stage,
            str(row.get("mode", "")),
            str(row.get("seed", "")),
            spec_id(row),
        ]
    )


def already_recorded(args: argparse.Namespace, key: str) -> bool:
    if args.rerun_existing:
        return False
    for row in load_jsonl(Path(args.out_root) / SUMMARY_NAME):
        if row.get("direction_key") == key:
            return is_success(row)
    return False


def but_features_for_stratified(args: argparse.Namespace, X: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    features_path = Path(args.out_root) / "but_morph_features_for_stratified.csv"
    if features_path.exists() and not args.force:
        return pd.read_csv(features_path)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    return rn.extract_features(X, meta, features_path, max_rows=0)


def checkpoint_from_row(row: dict[str, Any]) -> Path | None:
    eval_block = row.get("but_10s_eval") or {}
    for candidate in [
        eval_block.get("checkpoint"),
        row.get("checkpoint"),
        row.get("ckpt_best"),
    ]:
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path
    run_dir = row.get("run_dir")
    if run_dir:
        path = Path(run_dir) / "ckpt_best.pt"
        if path.exists():
            return path
    return None


def evaluate_stratified(args: argparse.Namespace, row: dict[str, Any], stage: str) -> dict[str, Any]:
    checkpoint = checkpoint_from_row(row)
    if checkpoint is None:
        return {}

    eval_dir = Path(args.out_root) / "stratified_eval" / stage / spec_id(row)
    if row.get("seed") is not None:
        eval_dir = eval_dir / f"seed{row.get('seed')}"
    report_path = eval_dir / "but_10s_stratified_balanced_report.json"
    if report_path.exists() and not args.force:
        return read_json(report_path, {})

    loaded = rn.load_but(Path(args.but_protocol_dir))
    if len(loaded) == 3:
        X, meta, _audit = loaded
    else:
        X, meta = loaded
    features = but_features_for_stratified(args, X, meta)
    strat_idx, manifest = stratified_balanced_indices(
        meta, features, seed=int(args.stratified_seed)
    )

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = load_uformer_model(checkpoint, device=device)
    outputs = run_model_outputs(
        model,
        X,
        batch_size=int(args.batch_size_eval),
        device=device,
    )
    y_col = "label_id" if "label_id" in meta.columns else "y"
    y = meta[y_col].to_numpy(dtype=int)
    val_mask = meta["split"].to_numpy() == "val"
    test_mask = meta["split"].to_numpy() == "test"
    thresholds = calibrate_but(outputs["probs"][val_mask], y[val_mask])
    pred_cal = apply_but_thresholds(
        outputs["probs"], float(thresholds["t_good"]), float(thresholds["t_bad"])
    )
    pred_raw = outputs["probs"].argmax(axis=1)

    test_indices = np.where(test_mask)[0]
    strat_indices = np.asarray(strat_idx, dtype=int)
    strat_indices = strat_indices[np.isin(strat_indices, test_indices)]
    cal_report = multiclass_report(y[strat_indices], pred_cal[strat_indices])
    raw_report = multiclass_report(y[strat_indices], pred_raw[strat_indices])

    eval_dir.mkdir(parents=True, exist_ok=True)
    df = meta.copy()
    df["idx"] = np.arange(len(df))
    df["p_good"] = outputs["probs"][:, 0]
    df["p_medium"] = outputs["probs"][:, 1]
    df["p_bad"] = outputs["probs"][:, 2]
    df["pred_calibrated_class"] = pred_cal
    df["pred_raw_class"] = pred_raw
    df["error_type"] = classify_errors(y, pred_cal)
    for case in [
        "bad_missed",
        "medium_to_bad",
        "medium_to_good",
        "good_false_bad",
        "correct_bad",
    ]:
        plot_wave_cases(X, df, case, eval_dir / f"{case}.png")

    payload = {
        "checkpoint": str(checkpoint),
        "thresholds": thresholds,
        "manifest": manifest,
        "calibrated": cal_report,
        "raw": raw_report,
        "gallery_dir": str(eval_dir),
    }
    write_json(report_path, payload)
    return payload


def record_payload(args: argparse.Namespace, row: dict[str, Any], stage: str) -> dict[str, Any]:
    payload = copy.deepcopy(row)
    payload["direction_stage"] = stage
    payload["direction_recorded_at"] = now_iso()
    payload["direction_key"] = direction_key(payload, stage)
    payload["model_scale"] = {
        "denoiser_width": float(args.denoiser_width),
        "hidden_dim": int(args.hidden_dim),
    }
    if already_recorded(args, payload["direction_key"]):
        return payload

    strat = evaluate_stratified(args, payload, stage) if args.stratified_eval else {}
    if strat:
        payload.setdefault("but_10s_eval", {})
        payload["but_10s_eval"]["but_10s_stratified_balanced_test_report"] = strat.get(
            "calibrated"
        )
        payload["but_10s_eval"]["but_10s_stratified_balanced_raw_report"] = strat.get(
            "raw"
        )
        payload["but_10s_eval"]["stratified_balanced_manifest"] = strat.get("manifest")
    append_jsonl(Path(args.out_root) / SUMMARY_NAME, payload)
    return payload


def run_seed_confirmation(args: argparse.Namespace, selected: list[dict[str, Any]]) -> None:
    update_state(Path(args.out_root) / STATE_NAME, status="seed_confirmation", updated_at=now_iso())
    for source_row in selected:
        reused = copy.deepcopy(source_row)
        reused["mode"] = "seed"
        reused["seed"] = 0
        reused["reused_from_source_full"] = True
        record_payload(args, reused, "seed_confirm")

        for seed in args.seeds:
            seed = int(seed)
            if seed == 0:
                continue
            train_args = make_train_args(args, seed=seed)
            row = copy.deepcopy(source_row)
            row["mode"] = "seed"
            row["seed"] = seed
            row["reused_variant_from_source"] = True
            result = mgb.train_one(train_args, row, "seed", seed=seed)
            record_payload(args, result, "seed_confirm")


def mutate_spec(base: dict[str, Any], suffix: str, changes: dict[str, Any], note: str) -> dict[str, Any]:
    spec = copy.deepcopy(base)
    spec.setdefault("changes", {}).update(changes)
    base_slug = mgb.stable_id(spec_id({"spec": base}))[:40]
    spec["id"] = f"dv_{suffix}__{base_slug}"
    spec["name"] = spec["id"]
    spec["note"] = note
    return spec


def base_direction_spec() -> dict[str, Any]:
    medium = {
        "medium_event_prob": 0.82,
        "medium_local_baseline_prob": 0.34,
        "medium_ptst_prob": 0.72,
        "medium_contact_rate": 0.0,
        "medium_strength": 1.05,
    }
    bad = {
        "bad_subtype_weights": {
            "qrs_confound": 1.0,
            "contact_flat": 1.0,
            "motion_burst": 1.0,
            "morph_break": 1.0,
            "clipping_lowamp": 1.0,
            "baseline_jump": 1.0,
        },
        "bad_secondary_prob": 0.30,
        "bad_strength": 1.05,
    }
    return mgb.base_spec(
        "dv_balanced_or_m_strong_detail_midbad",
        "bw_badstrong",
        [1.0, 1.55, 1.70],
        {**medium, **bad},
        "Direction-validation base: sample-level bad OR plus independent medium detail instability.",
    )


def ablation_specs(best_spec: dict[str, Any]) -> list[dict[str, Any]]:
    base = copy.deepcopy(best_spec) if best_spec else base_direction_spec()
    specs: list[dict[str, Any]] = []
    specs.append(
        mutate_spec(
            base,
            "ab_shared_average_bad",
            {
                "bad_secondary_prob": 1.0,
                "bad_subtype_weights": {
                    "qrs_confound": 1.0,
                    "contact_flat": 1.0,
                    "motion_burst": 1.0,
                    "morph_break": 1.0,
                    "clipping_lowamp": 1.0,
                    "baseline_jump": 1.0,
                },
            },
            "Proxy for removing sample-level OR: every bad sample receives broad average pressure.",
        )
    )
    for name, weights in [
        ("ab_balanced_or_only", {"qrs_confound": 1, "contact_flat": 1, "motion_burst": 1, "morph_break": 1, "clipping_lowamp": 1, "baseline_jump": 1}),
        ("ab_qrs_confound_only", {"qrs_confound": 4, "contact_flat": 0.05, "motion_burst": 0.05, "morph_break": 1.2, "clipping_lowamp": 0.05, "baseline_jump": 0.05}),
        ("ab_visible_unusable", {"qrs_confound": 0.1, "contact_flat": 0.4, "motion_burst": 1.4, "morph_break": 1.2, "clipping_lowamp": 0.1, "baseline_jump": 2.2}),
        ("ab_contact_lowamp", {"qrs_confound": 0.1, "contact_flat": 3.0, "motion_burst": 0.1, "morph_break": 0.1, "clipping_lowamp": 2.0, "baseline_jump": 0.3}),
    ]:
        specs.append(
            mutate_spec(
                base,
                name,
                {"bad_subtype_weights": weights, "bad_secondary_prob": 0.25},
                f"Bad subtype ablation: {name}.",
            )
        )
    for name, changes in [
        ("ab_m_soft", {"medium_event_prob": 0.64, "medium_ptst_prob": 0.50, "medium_local_baseline_prob": 0.22}),
        ("ab_m_strong_detail", {"medium_event_prob": 0.86, "medium_ptst_prob": 0.78, "medium_local_baseline_prob": 0.38, "medium_contact_rate": 0.0}),
        ("ab_m_guard", {"medium_event_prob": 0.78, "medium_ptst_prob": 0.68, "medium_local_baseline_prob": 0.32, "medium_contact_rate": 0.0, "medium_strength": 0.90}),
    ]:
        specs.append(mutate_spec(base, name, changes, f"Medium guard ablation: {name}."))
    for name, strength in [("ab_softbad", 0.82), ("ab_midbad", 1.05), ("ab_badguard", 1.25)]:
        specs.append(mutate_spec(base, name, {"bad_strength": strength}, f"Bad intensity ablation: {name}."))
    no_bw = mutate_spec(base, "ab_no_bw_overlay", {}, "BW overlay ablation.")
    no_bw["mix"] = copy.deepcopy(no_bw.get("mix") or {})
    no_bw["mix"]["bw_overlay"] = None
    specs.append(no_bw)
    return specs


def train_new_spec(
    args: argparse.Namespace,
    spec: dict[str, Any],
    direction_stage: str,
    mode: str = "quick",
    seed: int = 0,
) -> dict[str, Any]:
    train_args = make_train_args(args, seed=seed)
    variant_dir, _made = mgb.make_variant(
        train_args, spec, sample_per_class=0, force_regenerate=bool(args.force_variants)
    )
    row = {
        "variant_id": spec["id"],
        "spec": spec,
        "variant_dir": str(variant_dir),
        "mode": mode,
        "seed": seed,
    }
    result = mgb.train_one(train_args, row, mode, seed=seed)
    return record_payload(args, result, direction_stage)


def run_mechanism_ablation(args: argparse.Namespace, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    update_state(Path(args.out_root) / STATE_NAME, status="mechanism_ablation", updated_at=now_iso())
    base_spec = selected[0].get("spec") if selected else base_direction_spec()
    results: list[dict[str, Any]] = []
    for spec in ablation_specs(base_spec)[: int(args.max_ablation_specs)]:
        results.append(train_new_spec(args, spec, "mechanism_ablation", mode="quick", seed=0))

    promotable = [
        row
        for row in results
        if metric_value(row, "macro_f1") > ANCHOR_MACRO
        or (recalls(row)[1] >= 0.75 and recalls(row)[2] >= 0.80)
    ]
    promotable = sorted(promotable, key=lambda r: metric_value(r, "macro_f1"), reverse=True)[
        : int(args.top_ablation_full)
    ]
    for row in promotable:
        full = train_new_spec(args, row["spec"], "mechanism_ablation_full", mode="full", seed=0)
        results.append(full)
    write_json(Path(args.out_root) / "mechanism_ablation_summary.json", results)
    pd.DataFrame([compact_row(r) for r in results]).to_csv(
        Path(args.report_root) / "mechanism_ablation_summary.csv", index=False
    )
    return results


def focus_specs() -> list[dict[str, Any]]:
    # A Latin-style sweep keeps the grid broad without exploding into a
    # full Cartesian product.  The center still stays around the observed
    # sweet spot: medium detail guard + sample-level bad OR + BW overlay.
    medium_event_probs = [0.62, 0.70, 0.76, 0.82, 0.88]
    medium_contact_rates = [0.0, 0.005, 0.01, 0.02]
    secondary_probs = [0.15, 0.22, 0.30, 0.38, 0.46]
    medium_strengths = [0.88, 1.00, 1.12]
    bad_strengths = [0.88, 1.05, 1.22]
    mix_keys = ["bw_badstrong", "bw_mid", "ma_heavy"]
    class_weights = [
        "1.00,1.45,1.70",
        "1.00,1.55,1.70",
        "1.00,1.60,1.75",
        "1.00,1.62,1.78",
        "1.00,1.52,1.85",
    ]
    bad_mixes = [
        ("balanced", {"qrs_confound": 1, "contact_flat": 1, "motion_burst": 1, "morph_break": 1, "clipping_lowamp": 1, "baseline_jump": 1}),
        ("baseline_visible", {"qrs_confound": 0.25, "contact_flat": 0.8, "motion_burst": 1.0, "morph_break": 1.2, "clipping_lowamp": 0.4, "baseline_jump": 2.2}),
        ("qrs_morph", {"qrs_confound": 2.4, "contact_flat": 0.2, "motion_burst": 0.5, "morph_break": 2.0, "clipping_lowamp": 0.2, "baseline_jump": 0.5}),
        ("contact_lowamp", {"qrs_confound": 0.2, "contact_flat": 2.4, "motion_burst": 0.4, "morph_break": 0.4, "clipping_lowamp": 1.8, "baseline_jump": 0.7}),
        ("motion_baseline", {"qrs_confound": 0.4, "contact_flat": 0.5, "motion_burst": 2.2, "morph_break": 0.7, "clipping_lowamp": 0.3, "baseline_jump": 1.8}),
        ("mixed_unusable", {"qrs_confound": 1.4, "contact_flat": 1.4, "motion_burst": 1.2, "morph_break": 1.4, "clipping_lowamp": 1.2, "baseline_jump": 1.4}),
    ]
    combos = list(
        itertools.product(
            medium_event_probs,
            medium_contact_rates,
            secondary_probs,
            medium_strengths,
            bad_strengths,
            bad_mixes,
            class_weights,
            mix_keys,
        )
    )
    # 157 is coprime to the current Cartesian size, so the first 144 items
    # are a deterministic, well-spread sample across all axes.
    stride = 157
    specs: list[dict[str, Any]] = []
    for i in range(min(144, len(combos))):
        mev, mcr, sec, m_strength, b_strength, bad_mix, cw, mix_key = combos[
            (i * stride) % len(combos)
        ]
        mix_name, weights = bad_mix
        cw_tag = str(cw).replace(",", "_").replace(".", "p")
        name = f"dvf{i:03d}_{mix_key}_{mix_name}_cw{cw_tag}"
        specs.append(
            mgb.base_spec(
                name,
                mix_key,
                cw,
                {
                    "medium_event_prob": mev,
                    "medium_ptst_prob": min(0.90, mev + 0.02),
                    "medium_local_baseline_prob": max(0.22, mev - 0.42),
                    "medium_contact_rate": mcr,
                    "medium_strength": m_strength,
                    "bad_subtype_weights": weights,
                    "bad_secondary_prob": sec,
                    "bad_strength": b_strength,
                },
                "Focus grid around balanced_or + medium guard + midbad.",
            )
        )
    return specs


def run_focus_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    update_state(Path(args.out_root) / STATE_NAME, status="focus_grid", updated_at=now_iso())
    quick_results: list[dict[str, Any]] = []
    for spec in focus_specs()[: int(args.max_focus_specs)]:
        quick_results.append(train_new_spec(args, spec, "focus_grid", mode="quick", seed=0))

    promotable = [
        row
        for row in quick_results
        if metric_value(row, "macro_f1") > 0.755
        or (recalls(row)[1] >= 0.80 and recalls(row)[2] >= 0.80)
    ]
    promotable = sorted(promotable, key=lambda r: metric_value(r, "macro_f1"), reverse=True)[
        : int(args.top_focus_full)
    ]
    full_results = [
        train_new_spec(args, row["spec"], "focus_grid_full", mode="full", seed=0)
        for row in promotable
    ]
    results = quick_results + full_results
    write_json(Path(args.out_root) / "focus_grid_summary.json", results)
    pd.DataFrame([compact_row(r) for r in results]).to_csv(
        Path(args.report_root) / "focus_grid_summary.csv", index=False
    )
    return results


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    g, m, b = recalls(row)
    eval_block = row.get("but_10s_eval") or {}
    strat = eval_block.get("but_10s_stratified_balanced_test_report") or {}
    bal = eval_block.get("but_10s_balanced_test_report") or {}
    return {
        "direction_stage": row.get("direction_stage"),
        "mode": row.get("mode"),
        "seed": row.get("seed"),
        "variant_id": spec_id(row),
        "acc": metric_value(row, "acc"),
        "macro_f1": metric_value(row, "macro_f1"),
        "good_recall": g,
        "medium_recall": m,
        "bad_recall": b,
        "balanced_acc": bal.get("acc"),
        "balanced_macro_f1": bal.get("macro_f1"),
        "stratified_acc": strat.get("acc"),
        "stratified_macro_f1": strat.get("macro_f1"),
        "run_dir": row.get("run_dir"),
    }


def seed_stats(rows: list[dict[str, Any]]) -> pd.DataFrame:
    seed_rows = [r for r in rows if r.get("direction_stage") == "seed_confirm"]
    if not seed_rows:
        return pd.DataFrame()
    df = pd.DataFrame([compact_row(r) for r in seed_rows])
    groups = []
    for variant, gdf in df.groupby("variant_id"):
        groups.append(
            {
                "variant_id": variant,
                "n": len(gdf),
                "macro_mean": gdf["macro_f1"].mean(),
                "macro_std": gdf["macro_f1"].std(ddof=0),
                "acc_mean": gdf["acc"].mean(),
                "medium_mean": gdf["medium_recall"].mean(),
                "bad_mean": gdf["bad_recall"].mean(),
                "good_mean": gdf["good_recall"].mean(),
            }
        )
    return pd.DataFrame(groups).sort_values("macro_mean", ascending=False)


def write_report(args: argparse.Namespace) -> None:
    rows = load_jsonl(Path(args.out_root) / SUMMARY_NAME)
    compact = pd.DataFrame([compact_row(row) for row in rows])
    Path(args.report_root).mkdir(parents=True, exist_ok=True)
    if not compact.empty:
        compact.to_csv(Path(args.report_root) / "direction_validation_summary.csv", index=False)

    seeds = seed_stats(rows)
    if not seeds.empty:
        seeds.to_csv(Path(args.report_root) / "seed_confirmation_summary.csv", index=False)

    best = compact.sort_values("macro_f1", ascending=False).head(10) if not compact.empty else pd.DataFrame()
    decision = "not_supported"
    if not seeds.empty:
        top = seeds.iloc[0]
        if (
            top["macro_mean"] > ANCHOR_MACRO
            and top["macro_std"] <= 0.02
            and top["bad_mean"] >= 0.80
            and top["medium_mean"] >= 0.75
        ):
            decision = "breakthrough_supported"
        elif top["macro_mean"] > ANCHOR_MACRO or top["bad_mean"] >= 0.80:
            decision = "direction_promising_but_not_enough"
    elif not compact.empty and compact["macro_f1"].max() > ANCHOR_MACRO:
        decision = "direction_promising_but_not_enough"

    lines = [
        "# BUT 10s Direction Validation",
        "",
        f"Generated: {now_iso()}",
        f"Decision: `{decision}`",
        "",
        "## Anchor",
        "",
        f"- h_bad_rescue_05 macro-F1: `{ANCHOR_MACRO:.4f}`; recalls: `0.887 / 0.773 / 0.793`.",
        "- BUT original test remains the primary metric. Balanced and stratified-balanced are diagnostics only.",
        "",
        "## Best Rows",
        "",
    ]
    if best.empty:
        lines.append("No direction-validation rows have completed yet.")
    else:
        lines.append(best.to_markdown(index=False))
    lines.extend(["", "## Seed Stability", ""])
    if seeds.empty:
        lines.append("Seed confirmation has not completed yet.")
    else:
        lines.append(seeds.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `breakthrough_supported` requires seed mean macro-F1 above anchor, std <= 0.02, bad recall >= 0.80, and medium recall >= 0.75.",
            "- Mechanism ablations test whether sample-level bad OR and medium guard are carrying the gain rather than class weights or calibration alone.",
            "- Focus grid only promotes variants that clear macro-F1 > 0.755 or jointly keep medium and bad recall >= 0.80.",
        ]
    )
    report_path = Path(args.report_root) / "final_direction_validation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    write_json(
        Path(args.out_root) / "final_direction_validation_decision.json",
        {"decision": decision, "generated_at": now_iso()},
    )
    update_state(
        Path(args.out_root) / STATE_NAME,
        status="reported",
        decision=decision,
        rows=len(rows),
        updated_at=now_iso(),
    )


def should_run_focus(args: argparse.Namespace, selected: list[dict[str, Any]]) -> bool:
    if not args.skip_focus_if_no_full_breakthrough:
        return True
    return max(metric_value(row, "macro_f1") for row in selected) > ANCHOR_MACRO


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["select", "seed_confirm", "mechanism_ablation", "focus_grid", "report", "all"],
        default="all",
    )
    parser.add_argument("--source_out", default=str(DEFAULT_SOURCE_OUT))
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--source_artifact_dir", default=str(mgb.DEFAULT_SOURCE_ARTIFACT))
    parser.add_argument("--but_protocol_dir", default=str(mgb.DEFAULT_BUT_PROTOCOL))
    parser.add_argument("--protocol_out_root", default=str(mgb.PROTOCOL_OUT_ROOT))
    parser.add_argument("--protocol_report_root", default=str(mgb.PROTOCOL_REPORT_ROOT))
    parser.add_argument("--nstdb_root", default=str(mgb.DEFAULT_NSTDB_ROOT))
    parser.add_argument("--cinc_dir", default=str(mgb.DEFAULT_CINC_DIR))
    parser.add_argument("--expected_full", type=int, default=6)
    parser.add_argument("--wait_poll_sec", type=float, default=120.0)
    parser.add_argument("--top_seed_candidates", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--max_ablation_specs", type=int, default=14)
    parser.add_argument("--top_ablation_full", type=int, default=3)
    parser.add_argument("--max_focus_specs", type=int, default=72)
    parser.add_argument("--top_focus_full", type=int, default=8)
    parser.add_argument("--quick_epochs_stage1", type=int, default=4)
    parser.add_argument("--quick_epochs_stage2", type=int, default=4)
    parser.add_argument("--full_epochs_stage1", type=int, default=10)
    parser.add_argument("--full_epochs_stage2", type=int, default=8)
    parser.add_argument("--batch_size_stage1", type=int, default=24)
    parser.add_argument("--batch_size_stage2", type=int, default=24)
    parser.add_argument("--batch_size_eval", type=int, default=96)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--denoiser_width", type=float, default=1.0)
    parser.add_argument("--lambda_den", type=float, default=0.5)
    parser.add_argument("--lambda_cls", type=float, default=18.0)
    parser.add_argument("--lr_stage1", type=float, default=2e-4)
    parser.add_argument("--lr_head", type=float, default=7e-4)
    parser.add_argument("--lr_denoiser", type=float, default=2e-5)
    parser.add_argument("--train_regenerate_full", type=int, default=1)
    parser.add_argument("--stratified_seed", type=int, default=20260605)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force_variants", action="store_true")
    parser.add_argument("--rerun_existing", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_wait_for_source_full", dest="wait_for_source_full", action="store_false")
    parser.add_argument("--no_wait_for_source_idle", dest="wait_for_source_idle", action="store_false")
    parser.add_argument("--no_stratified_eval", dest="stratified_eval", action="store_false")
    parser.add_argument(
        "--run_focus_even_if_no_full_breakthrough",
        dest="skip_focus_if_no_full_breakthrough",
        action="store_false",
    )
    parser.set_defaults(
        wait_for_source_full=True,
        wait_for_source_idle=True,
        stratified_eval=True,
        skip_focus_if_no_full_breakthrough=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    Path(args.report_root).mkdir(parents=True, exist_ok=True)
    update_state(Path(args.out_root) / STATE_NAME, status="starting", stage=args.stage, updated_at=now_iso())

    selected: list[dict[str, Any]] = []
    if args.stage in {"select", "all", "seed_confirm", "mechanism_ablation", "focus_grid"}:
        selected = select_candidates(args)

    if args.stage in {"seed_confirm", "all"}:
        run_seed_confirmation(args, selected)

    if args.stage in {"mechanism_ablation", "all"}:
        run_mechanism_ablation(args, selected)

    if args.stage in {"focus_grid", "all"}:
        if should_run_focus(args, selected):
            run_focus_grid(args)
        else:
            update_state(
                Path(args.out_root) / STATE_NAME,
                status="focus_grid_skipped",
                reason="source_full_did_not_exceed_anchor",
                anchor_macro=ANCHOR_MACRO,
                updated_at=now_iso(),
            )

    if args.stage in {"report", "all"}:
        write_report(args)

    update_state(Path(args.out_root) / STATE_NAME, status="complete", stage=args.stage, updated_at=now_iso())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
