from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sqi_pipeline.config import SQIPipelineConfig
from src.sqi_pipeline.runner import ensure_dirs, load_step_callable, step_params, write_run_summary
from src.utils.data_downloads import ensure_sqi_raw_data

from src.supplemental_sqi_experiments.ablation_generalization import run_cross_noise_generalization, run_leave_one_out
from src.supplemental_sqi_experiments.common import project_root, validate_integrity, load_split_frame, write_json
from src.supplemental_sqi_experiments.final_claims import run_final_claims
from src.supplemental_sqi_experiments.fsqi_mechanism import run_fsqi_mechanism
from src.supplemental_sqi_experiments.mmd_paired_resampling import run_mmd_paired_resampling
from src.supplemental_sqi_experiments.model_diagnostics import run_model_diagnostics
from src.supplemental_sqi_experiments.model_seed_stability import run_model_seed_stability
from src.supplemental_sqi_experiments.noise_isolation import audit_noise_overlap, build_isolated_dataset
from src.supplemental_sqi_experiments.report import write_summary
from src.supplemental_sqi_experiments.stability import run_stability
from src.supplemental_sqi_experiments.strict_table6 import run_strict_table6


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fontTools").setLevel(logging.WARNING)


def default_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = project_root()
    out_root = root / "outputs" / "sqi_supplemental"
    report_root = root / "outputs" / "reports" / "sqi_supplemental"
    return {
        "paper_artifacts": Path(args.artifacts_dir or root / "outputs" / "sqi_paper_aligned"),
        "out_root": Path(args.out_dir or out_root),
        "report_root": Path(args.report_dir or report_root),
    }


def run_existing_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    paths = default_paths(args)
    out = paths["out_root"] / "existing_seed0"
    rep = paths["report_root"] / "existing_seed0"
    art = paths["paper_artifacts"]
    df = load_split_frame(art, normalized=True)
    integrity = validate_integrity(df)
    write_json(out / "integrity_summary.json", integrity)
    outputs: list[str] = [str(out / "integrity_summary.json")]
    outputs.extend(
        run_strict_table6(
            artifacts_dir=art,
            out_dir=out / "strict_table6",
            report_dir=rep / "strict_table6",
            seed=args.seed,
            C=args.C,
            gamma=args.gamma,
            force=args.force,
        )["outputs"]
    )
    outputs.extend(
        run_model_diagnostics(
            artifacts_dir=art,
            out_dir=out / "model_diagnostics",
            report_dir=rep / "model_diagnostics",
            n_boot=args.n_boot,
            seed=args.seed,
        )["outputs"]
    )
    outputs.extend(
        run_fsqi_mechanism(
            artifacts_dir=art,
            out_dir=out / "fsqi_mechanism",
            report_dir=rep / "fsqi_mechanism",
            max_records=args.max_records,
            force=args.force,
        )["outputs"]
    )
    outputs.extend(run_leave_one_out(artifacts_dir=art, out_dir=out / "ablation", report_dir=rep / "ablation", C=args.C, gamma=args.gamma, seed=args.seed)["outputs"])
    outputs.extend(
        run_cross_noise_generalization(
            artifacts_dir=art,
            out_dir=out / "generalization",
            report_dir=rep / "generalization",
            C=args.C,
            gamma=args.gamma,
            seed=args.seed,
        )["outputs"]
    )
    outputs.extend(
        run_final_claims(
            artifacts_dir=art,
            strict_dir=out / "strict_table6",
            out_dir=out / "final_claims",
            report_dir=rep / "final_claims",
            shared_images_dir=project_root() / "outputs" / "reports" / "sqi_paper_aligned" / "images",
            seed=args.seed,
            C=args.C,
            gamma=args.gamma,
            n_perm=args.n_perm,
        )["outputs"]
    )
    outputs.append(
        str(
            write_summary(
                out_root=out,
                report_root=rep,
                shared_images_dir=project_root() / "outputs" / "reports" / "sqi_paper_aligned" / "images",
            )
        )
    )
    return {"command": "diagnose-existing", "outputs": outputs}


def run_noise_audit(args: argparse.Namespace) -> dict[str, Any]:
    paths = default_paths(args)
    art = paths["paper_artifacts"]
    audit = art / "splits" / "split_seta_seed0_paper_balanced.audit.csv"
    split = art / "splits" / "split_seta_seed0_paper_balanced.csv"
    out = paths["out_root"] / "existing_seed0" / "noise_overlap_audit.csv"
    df = pd_read_csv_with_split(audit, split)
    temp = out.with_suffix(".tmp.csv")
    temp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(temp, index=False)
    overlap = audit_noise_overlap(temp, out)
    temp.unlink(missing_ok=True)
    return {"command": "audit-existing-noise", "outputs": [str(out)], "overlap_total": int(overlap["n_overlaps"].sum()) if not overlap.empty else 0}


def pd_read_csv_with_split(audit_csv: Path, split_csv: Path):
    import pandas as pd

    audit = pd.read_csv(audit_csv)
    split = pd.read_csv(split_csv, usecols=["record_id", "split"])
    audit["record_id"] = audit["record_id"].astype(str)
    split["record_id"] = split["record_id"].astype(str)
    audit = audit.merge(split, on="record_id", how="left")
    audit["noise_end_360"] = audit["noise_start_360"].astype(int) + 3600
    return audit


def build_isolated(args: argparse.Namespace) -> dict[str, Any]:
    root = project_root()
    paths = default_paths(args)
    art = paths["out_root"] / f"isolated_seed{args.seed}"
    cfg = SQIPipelineConfig.build(artifacts_dir=art, profile="paper_aligned", seed=args.seed, verbose=args.verbose, force=args.force)
    ensure_dirs(cfg.artifacts_dir)
    manifest_step = load_step_callable(next(s for s in __import__("src.sqi_pipeline.runner", fromlist=["PAPER_ALIGNED_STEPS"]).PAPER_ALIGNED_STEPS if s.name == "manifest_raw"))
    manifest_out = manifest_step(step_params(cfg, "manifest_raw"))
    params = {
        **cfg.base_params(),
        "manifest_csv": str(art / "manifests" / "manifest_challenge2011_seta.csv"),
        "out_split_csv": str(art / "splits" / f"split_seta_seed{args.seed}_paper_balanced.csv"),
        "audit_csv": str(art / "splits" / f"split_seta_seed{args.seed}_paper_balanced.audit.csv"),
        "overlap_csv": str(art / "splits" / f"split_seta_seed{args.seed}_paper_balanced.noise_overlap_audit.csv"),
        "qc_png": str(art / "qc" / f"isolated_paper_balanced_seed{args.seed}_label_counts.png"),
        "set_a_dir": str(cfg.set_a_dir),
        "nstdb_dir": str(cfg.nstdb_root),
        "cases_500_dir": str(art / "cases_500"),
        "noise_start_stride_s": args.noise_stride_s,
        "force": args.force,
        "seed": args.seed,
    }
    isolated_out = build_isolated_dataset(params)
    return {"command": "build-isolated", "outputs": manifest_out.get("outputs", []) + isolated_out.get("outputs", [])}


def run_isolated_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    # This intentionally uses the main pipeline modules after the isolated split/cases
    # are created, but writes everything under outputs/sqi_supplemental.
    paths = default_paths(args)
    art = paths["out_root"] / f"isolated_seed{args.seed}"
    build_isolated(args)
    cfg = SQIPipelineConfig.build(artifacts_dir=art, profile="paper_aligned", seed=args.seed, verbose=args.verbose, force=args.force)
    ensure_dirs(cfg.artifacts_dir)
    steps = ["resample_125", "qrs_cache", "record84", "norm_record84_ks", "lm_mlp_search", "svm_tables"]
    summary = {"profile": "paper_aligned_isolated_supplemental", "seed": args.seed, "artifacts_dir": str(art), "steps": []}
    for step_name in steps:
        fn = load_step_callable(next(s for s in __import__("src.sqi_pipeline.runner", fromlist=["PAPER_ALIGNED_STEPS"]).PAPER_ALIGNED_STEPS if s.name == step_name))
        params = step_params(cfg, step_name)
        if step_name in {"resample_125", "qrs_cache", "record84", "norm_record84_ks", "lm_mlp_search", "svm_tables"}:
            old = art / "splits" / f"split_seta_seed{args.seed}_paper_balanced.csv"
            params["split_csv"] = str(old)
        if step_name == "paper_balanced_seta":
            continue
        out = fn(params)
        summary["steps"].append({"name": step_name, "outputs": out.get("outputs", []), "skipped": bool(out.get("skipped", False))})
    summary_path = write_run_summary(summary, cfg)
    return {"command": "run-isolated", "outputs": [str(summary_path)]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supplemental protocol experiments for the SQI paper-aligned reproduction.")
    parser.add_argument("command", choices=["diagnose-existing", "strict-table6", "model-diagnostics", "fsqi", "ablation", "generalization", "final-claims", "stability", "model-seed-stability", "mmd-paired-resampling", "audit-existing-noise", "build-isolated", "run-isolated"])
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.14)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--max-records", type=int, default=None, help="debug limiter per sample group for fSQI mechanism")
    parser.add_argument("--n-perm", type=int, default=1000, help="permutations for RBF-MMD domain-shift test")
    parser.add_argument("--noise-stride-s", type=float, default=1.0)
    parser.add_argument("--split-seeds", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
    parser.add_argument("--mlp-init-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--model-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--mmd-resamples", type=int, default=1000)
    parser.add_argument("--include-mlp", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    ensure_sqi_raw_data(project_root())
    paths = default_paths(args)
    if args.command == "diagnose-existing":
        out = run_existing_diagnostics(args)
    elif args.command == "strict-table6":
        out = run_strict_table6(artifacts_dir=paths["paper_artifacts"], out_dir=paths["out_root"] / "existing_seed0" / "strict_table6", report_dir=paths["report_root"] / "existing_seed0" / "strict_table6", seed=args.seed, C=args.C, gamma=args.gamma, force=args.force)
    elif args.command == "model-diagnostics":
        out = run_model_diagnostics(artifacts_dir=paths["paper_artifacts"], out_dir=paths["out_root"] / "existing_seed0" / "model_diagnostics", report_dir=paths["report_root"] / "existing_seed0" / "model_diagnostics", n_boot=args.n_boot, seed=args.seed)
    elif args.command == "fsqi":
        out = run_fsqi_mechanism(artifacts_dir=paths["paper_artifacts"], out_dir=paths["out_root"] / "existing_seed0" / "fsqi_mechanism", report_dir=paths["report_root"] / "existing_seed0" / "fsqi_mechanism", max_records=args.max_records, force=args.force)
    elif args.command == "ablation":
        out = run_leave_one_out(artifacts_dir=paths["paper_artifacts"], out_dir=paths["out_root"] / "existing_seed0" / "ablation", report_dir=paths["report_root"] / "existing_seed0" / "ablation", C=args.C, gamma=args.gamma, seed=args.seed)
    elif args.command == "generalization":
        out = run_cross_noise_generalization(artifacts_dir=paths["paper_artifacts"], out_dir=paths["out_root"] / "existing_seed0" / "generalization", report_dir=paths["report_root"] / "existing_seed0" / "generalization", C=args.C, gamma=args.gamma, seed=args.seed)
    elif args.command == "final-claims":
        strict_dir = paths["out_root"] / "existing_seed0" / "strict_table6"
        if not (strict_dir / "all_127_subset_val.csv").exists():
            run_strict_table6(
                artifacts_dir=paths["paper_artifacts"],
                out_dir=strict_dir,
                report_dir=paths["report_root"] / "existing_seed0" / "strict_table6",
                seed=args.seed,
                C=args.C,
                gamma=args.gamma,
                force=args.force,
            )
        out = run_final_claims(
            artifacts_dir=paths["paper_artifacts"],
            strict_dir=strict_dir,
            out_dir=paths["out_root"] / "existing_seed0" / "final_claims",
            report_dir=paths["report_root"] / "existing_seed0" / "final_claims",
            shared_images_dir=project_root() / "outputs" / "reports" / "sqi_paper_aligned" / "images",
            seed=args.seed,
            C=args.C,
            gamma=args.gamma,
            n_perm=args.n_perm,
        )
    elif args.command == "stability":
        split_seeds = [int(x) for x in str(args.split_seeds).split(",") if x.strip()]
        mlp_init_seeds = [int(x) for x in str(args.mlp_init_seeds).split(",") if x.strip()]
        out = run_stability(
            artifacts_dir=paths["paper_artifacts"],
            out_dir=paths["out_root"] / "existing_seed0" / "stability",
            report_dir=paths["report_root"] / "existing_seed0" / "stability",
            split_seeds=split_seeds,
            mlp_init_seeds=mlp_init_seeds,
            include_mlp=bool(args.include_mlp),
        )
    elif args.command == "model-seed-stability":
        model_seeds = [int(x) for x in str(args.model_seeds).split(",") if x.strip()]
        out = run_model_seed_stability(
            artifacts_dir=paths["paper_artifacts"],
            out_dir=paths["out_root"] / "existing_seed0" / "model_seed_stability",
            model_seeds=model_seeds,
        )
    elif args.command == "mmd-paired-resampling":
        out = run_mmd_paired_resampling(
            artifacts_dir=paths["paper_artifacts"],
            out_dir=paths["out_root"] / "existing_seed0" / "mmd_paired_resampling",
            n_resamples=args.mmd_resamples,
            seed=args.seed,
        )
    elif args.command == "audit-existing-noise":
        out = run_noise_audit(args)
    elif args.command == "build-isolated":
        out = build_isolated(args)
    elif args.command == "run-isolated":
        out = run_isolated_pipeline(args)
    else:
        raise ValueError(args.command)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
