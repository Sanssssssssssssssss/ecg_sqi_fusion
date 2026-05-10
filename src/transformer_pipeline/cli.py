from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transformer_pipeline.config import TransformerPipelineConfig
from src.transformer_pipeline.runner import ensure_dirs, format_summary_table, fresh_artifacts, run_pipeline, write_run_summary


def parse_args(*, default_stage: str = "all") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PTB-XL transformer pipeline.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="delete generated transformer artifacts and rerun")
    parser.add_argument("--force", action="store_true", help="force each step to rerun")
    parser.add_argument("--dry-run", action="store_true", help="for train: load data and run one forward pass, no training")
    parser.add_argument("--stage", choices=("all", "preprocess", "model"), default=default_stage)
    parser.add_argument("--only", default="", help="comma-separated step names, e.g. forward_check,train")
    parser.add_argument("--artifact_dir", default="outputs/transformer")
    parser.add_argument("--source_artifact_dir", default="", help="read source segments/splits from this artifact dir")
    parser.add_argument("--preserve_eval_from", default="", help="copy val/test synthetic data from this artifact dir")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_aug_mode", choices=("single", "multiview", "triplet"), default=None)
    parser.add_argument("--train_aug_k", type=int)
    parser.add_argument("--train_noise_kinds", default="")
    parser.add_argument("--stratify_noise_snr", action="store_true")
    parser.add_argument("--experiment_name", default="", help="model subdirectory under artifact_dir/models")
    parser.add_argument("--model_dir", default="", help="explicit model output directory")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr_eta_min", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--cls_pool", choices=("decoder", "encoder", "both"), default=None)
    parser.add_argument("--input_mode", choices=("raw", "robust", "raw_robust"), default=None)
    parser.add_argument("--ordinal_head", action="store_true")
    parser.add_argument("--snr_head", action="store_true")
    parser.add_argument("--e_cls", type=int)
    parser.add_argument("--e_denoise", type=int)
    parser.add_argument("--e_level", type=int)
    parser.add_argument("--e_uncert", type=int)
    parser.add_argument("--bad_den_w_max", type=float)
    parser.add_argument("--bad_den_w_warmup_epochs", type=int)
    parser.add_argument("--lambda_cls", type=float)
    parser.add_argument("--lambda_den", type=float)
    parser.add_argument("--lambda_lvl", type=float)
    parser.add_argument("--lambda_ord", type=float)
    parser.add_argument("--lambda_snr", type=float)
    parser.add_argument("--label_smoothing", type=float)
    parser.add_argument("--class_weight_good", type=float)
    parser.add_argument("--class_weight_medium", type=float)
    parser.add_argument("--class_weight_bad", type=float)
    parser.add_argument("--select_best_by", choices=("val_acc", "val_loss"), default=None)
    parser.add_argument("--uncertainty_mode", choices=("kendall", "fixed"), default=None)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--earlystop_patience", type=int)
    parser.add_argument("--earlystop_min_delta", type=float)
    parser.add_argument("--earlystop_start_epoch", type=int)
    return parser.parse_args()


def train_overrides_from_args(args: argparse.Namespace) -> dict[str, object]:
    keys = (
        "experiment_name",
        "model_dir",
        "source_artifact_dir",
        "preserve_eval_from",
        "train_aug_mode",
        "train_aug_k",
        "train_noise_kinds",
        "stratify_noise_snr",
        "epochs",
        "batch_size",
        "num_workers",
        "pin_memory",
        "lr",
        "lr_eta_min",
        "weight_decay",
        "dropout",
        "cls_pool",
        "input_mode",
        "ordinal_head",
        "snr_head",
        "e_cls",
        "e_denoise",
        "e_level",
        "e_uncert",
        "bad_den_w_max",
        "bad_den_w_warmup_epochs",
        "lambda_cls",
        "lambda_den",
        "lambda_lvl",
        "lambda_ord",
        "lambda_snr",
        "label_smoothing",
        "class_weight_good",
        "class_weight_medium",
        "class_weight_bad",
        "select_best_by",
        "uncertainty_mode",
        "early_stop",
        "earlystop_patience",
        "earlystop_min_delta",
        "earlystop_start_epoch",
    )
    out: dict[str, object] = {}
    for key in keys:
        value = getattr(args, key)
        if value is None:
            continue
        if isinstance(value, str) and not value:
            continue
        if isinstance(value, bool) and not value:
            continue
        out[key] = value
    return out


def setup_logging(verbose: bool) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)


def main(*, default_stage: str = "all") -> None:
    args = parse_args(default_stage=default_stage)
    setup_logging(args.verbose)

    cfg = TransformerPipelineConfig.build(
        artifact_dir=args.artifact_dir,
        seed=args.seed,
        verbose=args.verbose,
        dry_run=args.dry_run,
        force=args.force,
        train_overrides=train_overrides_from_args(args),
    )
    if args.fresh:
        fresh_artifacts(cfg.artifact_dir, stage=args.stage)
    ensure_dirs(cfg.artifact_dir)

    only = [s.strip() for s in args.only.split(",") if s.strip()] or None
    if args.dry_run and args.stage in {"all", "model"} and only is None:
        only = ["forward_check", "train"]
    summary = run_pipeline(cfg, only=only, stage=args.stage)
    out = write_run_summary(summary, cfg)
    logging.getLogger(__name__).info("Run summary written: %s", out)
    print(format_summary_table(summary, title=f"Transformer {args.stage} summary"))


if __name__ == "__main__":
    main()
