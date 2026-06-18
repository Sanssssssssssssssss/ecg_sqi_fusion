"""Fit PTB synthetic rules to CleanBUT-Core targets before BUT testing.

Stage 0 ranks medium-guard/Sensors-style PTB synthetic variants by distance to
the CleanBUT-Core target distributions.  Later stages reuse the existing
Uformer training/evaluation path and still report the original BUT 10s P1 test
as the primary metric.  CleanBUT is used only as a diagnostic/generator target.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.transformer_pipeline.e311_uformer_eval import write_json
from src.transformer_pipeline.external_benchmarks import but_clean_label_core_atlas_10s as clean
from src.transformer_pipeline.external_benchmarks import but_medium_guard_bad_boundary_grid_10s as mg
from src.transformer_pipeline.external_benchmarks import real_noise_snr_but_match_grid_10s as rn
from src.transformer_pipeline.external_benchmarks.but_bad_boundary_tuning import (
    DEFAULT_SOURCE_ARTIFACT,
    now_iso,
    read_json,
    update_state,
)


ROOT = mg.ROOT
RUN_TAG = "e311_but_clean_target_fit_grid_10s_2026_06_06"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "external_benchmarks" / RUN_TAG
DEFAULT_REPORT_ROOT = ROOT / "reports" / "external_benchmarks" / RUN_TAG
DEFAULT_CLEAN_ROOT = clean.DEFAULT_OUT_ROOT
STATE_NAME = "clean_target_fit_state.json"
SPEC_NAME = "clean_target_fit_specs.json"
SCAN_NAME = "clean_target_fit_distance_leaderboard.csv"
SUMMARY_NAME = "clean_target_fit_training_summary.jsonl"
CLASS_NAMES = clean.CLASS_NAMES


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def clean_target_features(clean_root: Path) -> tuple[pd.DataFrame, list[str]]:
    manifest_path = clean_root / "clean_subset_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"CleanBUT manifest missing: {manifest_path}. Run but_clean_label_core_atlas_10s first.")
    df = pd.read_csv(manifest_path)
    target = df[df["is_clean_train_target"].astype(bool)].copy()
    features = [f for f in rn.FEATURE_COLUMNS if f in target.columns]
    if len(features) < 8:
        features = [
            c
            for c in target.select_dtypes(include=[np.number]).columns
            if c not in clean.atlas.NON_FEATURE_COLUMNS and not c.startswith("pc")
        ][:32]
    if len(features) < 8:
        raise ValueError(f"Not enough common CleanBUT/rn feature columns: {len(features)}")
    return target, features


def clean_distance(clean_target: pd.DataFrame, synth_feat: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    common = [f for f in features if f in synth_feat.columns]
    dist = clean.classwise_distance(clean_target, synth_feat, common)
    domain_bal = rn.domain_separability(clean_target, synth_feat, common, seed=20260606) if len(common) >= 6 else math.nan
    medium = float(dist["by_class"].get("medium", math.nan))
    bad = float(dist["by_class"].get("bad", math.nan))
    good = float(dist["by_class"].get("good", math.nan))
    overall = float(dist["overall"])
    score = float(0.34 * overall + 0.28 * medium + 0.24 * bad + 0.08 * good + 0.06 * domain_bal)
    return {
        "clean_target_score": score,
        "clean_overall_distance": overall,
        "clean_good_distance": good,
        "clean_medium_distance": medium,
        "clean_bad_distance": bad,
        "clean_domain_separability_bal_acc": domain_bal,
        "n_common_features": int(len(common)),
        "rows": dist["rows"],
    }


def make_specs(max_specs: int) -> list[dict[str, Any]]:
    specs = mg.make_specs(max_specs)
    for spec in specs:
        spec["family"] = "clean_target_fit_medium_guard"
        spec["selection_basis"] = "CleanBUT-Core class-wise morphology distance; no BUT test metric used for scan ranking."
    return specs


def score_one(args: argparse.Namespace, spec: dict[str, Any], clean_target: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    variant_dir, made = mg.make_variant(args, spec, int(args.scan_sample_per_class))
    report_path = variant_dir / "clean_target_distance_report.json"
    if report_path.exists() and not args.force:
        return read_json(report_path)
    X_synth = np.load(variant_dir / "datasets" / "signals.npz")["X"].astype(np.float32)
    labels = pd.read_csv(variant_dir / "datasets" / "synth_10s_125hz_labels_with_level.csv")
    meta = labels.copy()
    meta["class_name"] = meta["y_class"].astype(str)
    synth_feat = rn.extract_features(X_synth, meta, variant_dir / "synthetic_morph_features.csv", max_rows=0)
    dist = clean_distance(clean_target, synth_feat, features)
    row = {
        "variant_id": str(spec["id"]),
        "family": spec.get("family", "clean_target_fit_medium_guard"),
        "spec": spec,
        "variant_dir": str(variant_dir),
        "audit": made.get("audit", {}),
        **{k: v for k, v in dist.items() if k != "rows"},
    }
    write_json(variant_dir / "clean_target_distance_report.json", row)
    write_json(variant_dir / "clean_target_distance_rows.json", dist["rows"])
    write_json(
        variant_dir / "synthetic_feature_profile.json",
        {"morph_features": synth_feat.describe().to_dict(), "audit": made.get("audit", {}), "clean_target_distance": row},
    )
    return row


def score_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    update_state(out_root / STATE_NAME, status="scan_running", stage="scan", updated_at=now_iso())
    clean_target, features = clean_target_features(Path(args.clean_root))
    specs = make_specs(int(args.max_specs))
    write_json(out_root / SPEC_NAME, {"rows": specs})
    write_json(report_root / SPEC_NAME, {"rows": specs})
    rows: list[dict[str, Any]] = []
    for i, spec in enumerate(specs, start=1):
        update_state(out_root / STATE_NAME, status="scan_running", current=spec["id"], completed=i - 1, total=len(specs), updated_at=now_iso())
        rows.append(score_one(args, spec, clean_target, features))
    rows = sorted(rows, key=lambda r: float(r["clean_target_score"]))
    write_json(out_root / "clean_target_fit_distance_leaderboard.json", {"rows": rows})
    df = pd.DataFrame(
        [
            {
                "rank": i,
                "variant_id": r["variant_id"],
                "clean_target_score": r["clean_target_score"],
                "clean_overall_distance": r["clean_overall_distance"],
                "clean_good_distance": r["clean_good_distance"],
                "clean_medium_distance": r["clean_medium_distance"],
                "clean_bad_distance": r["clean_bad_distance"],
                "clean_domain_separability_bal_acc": r["clean_domain_separability_bal_acc"],
                "n_common_features": r["n_common_features"],
                "variant_dir": r["variant_dir"],
            }
            for i, r in enumerate(rows, start=1)
        ]
    )
    df.to_csv(out_root / SCAN_NAME, index=False)
    df.to_csv(report_root / SCAN_NAME, index=False)
    update_state(out_root / STATE_NAME, status="scan_complete", stage="scan", completed=len(rows), total=len(rows), updated_at=now_iso())
    write_report(args)
    return rows


def select_for_quick(args: argparse.Namespace) -> list[dict[str, Any]]:
    ranked_path = Path(args.out_root) / "clean_target_fit_distance_leaderboard.json"
    rows = read_json(ranked_path)["rows"] if ranked_path.exists() else score_specs(args)
    selected = rows[: int(args.top_quick)]
    by_id = {str(r["variant_id"]): r for r in rows}
    for control in str(args.control_variant_ids).split(","):
        control = control.strip()
        if control and control in by_id and by_id[control] not in selected:
            selected.append(by_id[control])
    return selected[: int(args.top_quick) + 4]


def train_one(args: argparse.Namespace, row: dict[str, Any], mode: str, seed: int = 0) -> dict[str, Any]:
    result = mg.train_one(args, row, mode, seed=seed)
    result["clean_target_distance"] = {
        k: row.get(k)
        for k in [
            "clean_target_score",
            "clean_overall_distance",
            "clean_good_distance",
            "clean_medium_distance",
            "clean_bad_distance",
            "clean_domain_separability_bal_acc",
        ]
    }
    append_jsonl(Path(args.out_root) / SUMMARY_NAME, result)
    write_report(args)
    return result


def run_quick(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = select_for_quick(args)
    update_state(Path(args.out_root) / STATE_NAME, status="quick_training", stage="quick", total=len(selected), updated_at=now_iso())
    rows = []
    for i, row in enumerate(selected, start=1):
        update_state(Path(args.out_root) / STATE_NAME, status="quick_training", stage="quick", current=row["variant_id"], completed=i - 1, total=len(selected), updated_at=now_iso())
        rows.append(train_one(args, row, "quick", seed=int(args.seed)))
    update_state(Path(args.out_root) / STATE_NAME, status="quick_complete", stage="quick", completed=len(rows), total=len(rows), updated_at=now_iso())
    return rows


def _metric_score(row: dict[str, Any]) -> tuple[float, float, float, float]:
    rep = row.get("but_10s_eval", {}).get("but_10s_test_report", {})
    rec = rep.get("recall_good_medium_bad", [0.0, 0.0, 0.0])
    return (float(rep.get("macro_f1", 0.0)), min(float(rec[1]), float(rec[2])), float(rep.get("acc", 0.0)), float(rec[2]))


def run_full(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = load_jsonl(Path(args.out_root) / SUMMARY_NAME)
    quick = [r for r in rows if r.get("mode") == "quick" and int(r.get("returncode", 1)) == 0]
    if not quick:
        quick = run_quick(args)
    selected = sorted(quick, key=_metric_score, reverse=True)[: int(args.top_full)]
    update_state(Path(args.out_root) / STATE_NAME, status="full_training", stage="full", total=len(selected), updated_at=now_iso())
    out = []
    for i, row in enumerate(selected, start=1):
        update_state(Path(args.out_root) / STATE_NAME, status="full_training", stage="full", current=row["spec"]["id"], completed=i - 1, total=len(selected), updated_at=now_iso())
        out.append(train_one(args, row, "full", seed=int(row.get("seed", args.seed))))
    update_state(Path(args.out_root) / STATE_NAME, status="full_complete", stage="full", completed=len(out), total=len(out), updated_at=now_iso())
    return out


def write_report(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    scan = pd.read_csv(out_root / SCAN_NAME) if (out_root / SCAN_NAME).exists() else pd.DataFrame()
    rows = load_jsonl(out_root / SUMMARY_NAME)
    train_rows = []
    for r in rows:
        rep = r.get("but_10s_eval", {}).get("but_10s_test_report", {})
        rec = rep.get("recall_good_medium_bad", [math.nan, math.nan, math.nan])
        train_rows.append(
            {
                "mode": r.get("mode"),
                "variant_id": r.get("spec", {}).get("id", r.get("variant_id")),
                "acc": rep.get("acc"),
                "macro_f1": rep.get("macro_f1"),
                "good_recall": rec[0],
                "medium_recall": rec[1],
                "bad_recall": rec[2],
                **r.get("clean_target_distance", {}),
            }
        )
    train_df = pd.DataFrame(train_rows)
    if not train_df.empty:
        train_df.to_csv(out_root / "clean_target_fit_metric_join.csv", index=False)
        train_df.to_csv(report_root / "clean_target_fit_metric_join.csv", index=False)
    lines = [
        "# BUT Clean Target Fit Grid",
        "",
        "## Scan leaderboard",
        "",
        clean.md_table(scan.head(20), 20) if not scan.empty else "_Scan has not run._",
        "",
        "## Training results",
        "",
        clean.md_table(train_df.sort_values("macro_f1", ascending=False).head(20), 20) if not train_df.empty else "_Training has not run._",
        "",
        "## Notes",
        "",
        "- Stage 0 ranking uses CleanBUT-Core class-wise distance only.",
        "- Original BUT 10s P1 remains the primary test metric after training.",
        "- CleanBUT is a generator target and diagnostic subset, not a replacement benchmark.",
    ]
    (report_root / "clean_target_fit_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_all(args: argparse.Namespace) -> None:
    score_specs(args)
    if int(args.run_training) == 1:
        run_quick(args)
        run_full(args)
        update_state(Path(args.out_root) / STATE_NAME, status="complete", stage="all", updated_at=now_iso())


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["scan", "quick", "full", "report", "all"], default="scan")
    p.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--report_root", default=str(DEFAULT_REPORT_ROOT))
    p.add_argument("--clean_root", default=str(DEFAULT_CLEAN_ROOT))
    p.add_argument("--source_artifact_dir", default=str(DEFAULT_SOURCE_ARTIFACT))
    p.add_argument("--but_protocol_dir", default=str(mg.DEFAULT_BUT_PROTOCOL))
    p.add_argument("--nstdb_root", default=str(mg.DEFAULT_NSTDB_ROOT))
    p.add_argument("--cinc_dir", default=str(mg.DEFAULT_CINC_DIR))
    p.add_argument("--max_specs", type=int, default=36)
    p.add_argument("--scan_sample_per_class", type=int, default=60)
    p.add_argument("--top_quick", type=int, default=12)
    p.add_argument("--top_full", type=int, default=4)
    p.add_argument("--control_variant_ids", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--rerun_existing", action="store_true")
    p.add_argument("--skip_failed_existing", action="store_true")
    p.add_argument("--run_training", type=int, default=0)
    p.add_argument("--train_regenerate_full", type=int, default=1)
    p.add_argument("--quick_epochs_stage1", type=int, default=5)
    p.add_argument("--quick_epochs_stage2", type=int, default=5)
    p.add_argument("--full_epochs_stage1", type=int, default=14)
    p.add_argument("--full_epochs_stage2", type=int, default=10)
    p.add_argument("--batch_size_stage1", type=int, default=16)
    p.add_argument("--batch_size_stage2", type=int, default=24)
    p.add_argument("--batch_size_eval", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--denoiser_width", type=float, default=1.5)
    p.add_argument("--lambda_den_stage2", type=float, default=0.5)
    p.add_argument("--lambda_cls", type=float, default=18.0)
    p.add_argument("--class_weight", default="1.00,1.55,1.70")
    p.add_argument("--cpu", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    Path(args.report_root).mkdir(parents=True, exist_ok=True)
    if args.stage == "scan":
        score_specs(args)
    elif args.stage == "quick":
        run_quick(args)
    elif args.stage == "full":
        run_full(args)
    elif args.stage == "report":
        write_report(args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
