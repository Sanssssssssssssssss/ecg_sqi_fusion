from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


THIS_FILE = Path(__file__).absolute()
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else next(p for p in THIS_FILE.parents if (p / "src").exists())
EXP = ROOT / "outputs" / "experiment" / "e311_sqi_denoise_classifier_grid"
SCRIPT = EXP / "encoder_head_audit.py"
PURE_TRAIN = ROOT / "src" / "experiment" / "e311_sqi_research" / "train.py"
ARTIFACT = EXP / "artifact" / "transformer_reentry_audit"
LOG_DIR = EXP / "logs" / "transformer_reentry_audit"
STATE = EXP / "transformer_reentry_state.json"
SUMMARY = EXP / "transformer_reentry_summary.jsonl"
SPECS = EXP / "transformer_reentry_specs.jsonl"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
SOURCE = ROOT / "outputs" / "experiment" / "e311_morph_denoise_gap5_7_grid" / "data" / "med6p25_badgap7_badcm0p75"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tag(value: float | str) -> str:
    return str(value).replace(".", "p").replace("-", "m").replace(",", "c")


def encoder_base_args(out_dir: Path, variant: str, seed: int, init_mode: str) -> list[str]:
    return [
        str(PYTHON),
        str(SCRIPT),
        "--variant",
        variant,
        "--output_dir",
        str(out_dir),
        "--source_artifact_dir",
        str(SOURCE),
        "--feature_set",
        "full",
        "--epochs",
        "8",
        "--batch_size",
        "56",
        "--lambda_cls",
        "18",
        "--lambda_den",
        "0",
        "--lambda_snr",
        "0.03",
        "--lambda_local",
        "0.005",
        "--class_weight",
        "1,1.40,1.70",
        "--lr_denoiser",
        "0",
        "--lr_classifier",
        "2e-5",
        "--lr_head",
        "7e-4",
        "--seed",
        str(seed),
        "--init_mode",
        init_mode,
        "--freeze_denoiser",
        "--detach_encoder_features",
    ]


def make_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    for init_mode, label in [("warm", "warmcls"), ("denoiser_warm_classifier_he", "clshe")]:
        for seed in [0, 1, 2]:
            out = ARTIFACT / f"r1_tf_no_sqi_{label}_seed{seed}"
            specs.append(
                {
                    "stage": "round1_transformer_baseline",
                    "kind": "encoder_audit",
                    "run_id": out.name,
                    "seed": seed,
                    "variant": "dual_branch",
                    "init_mode": init_mode,
                    "command": [
                        *encoder_base_args(out, "dual_branch", seed, init_mode),
                        "--delta_scale",
                        "0.0",
                    ],
                }
            )

    for seed in [0, 1, 2]:
        out = ARTIFACT / f"r1_tf_latent_delta_warm_seed{seed}"
        specs.append(
            {
                "stage": "round1_transformer_latent_delta",
                "kind": "encoder_audit",
                "run_id": out.name,
                "seed": seed,
                "variant": "dual_branch",
                "init_mode": "warm",
                "command": [
                    *encoder_base_args(out, "dual_branch", seed, "warm"),
                    "--gated_delta",
                    "--delta_scale",
                    "0.04",
                ],
            }
        )

    for variant in ["transformer_film_sqi", "transformer_cross_sqi", "transformer_sqi_token_prefix"]:
        for init_mode, label in [("warm", "warmcls"), ("denoiser_warm_classifier_he", "clshe")]:
            for seed in [0, 1]:
                out = ARTIFACT / f"r2_{variant}_{label}_seed{seed}"
                specs.append(
                    {
                        "stage": "round2_transformer_latent_adapter",
                        "kind": "encoder_audit",
                        "run_id": out.name,
                        "seed": seed,
                        "variant": variant,
                        "init_mode": init_mode,
                        "command": [
                            *encoder_base_args(out, variant, seed, init_mode),
                            "--delta_scale",
                            "0.10",
                        ],
                    }
                )

    pure_out = ARTIFACT / "r3_pure_transformer"
    for recipe in ["hr_baseline_clone", "sqi_mil_detach_den10_lvl025"]:
        specs.append(
            {
                "stage": "round3_pure_transformer",
                "kind": "pure_transformer",
                "run_id": f"r3_pure_transformer_{recipe}",
                "seed": 1,
                "variant": recipe,
                "init_mode": "scratch_or_missing_warm",
                "command": [
                    str(PYTHON),
                    str(PURE_TRAIN),
                    "--group",
                    "head_reimpl" if recipe == "hr_baseline_clone" else "sqi_head_tuning",
                    "--recipe",
                    recipe,
                    "--artifact_dir",
                    str(SOURCE),
                    "--out_root",
                    str(pure_out),
                    "--init_checkpoint",
                    str(ARTIFACT / "missing_pure_transformer_scratch.pt"),
                ],
            }
        )
    return specs


def summarize_encoder_run(spec: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    row = {
        "stage": spec["stage"],
        "kind": spec["kind"],
        "run_id": spec["run_id"],
        "seed": spec["seed"],
        "variant": spec["variant"],
        "init_mode": spec["init_mode"],
        "output_dir": str(out_dir),
    }
    summary = out_dir / "run_summary.json"
    if summary.exists():
        data = read_json(summary)
        report = data.get("test_report", {})
        metrics = data.get("denoise_metrics", {})
        recalls = report.get("final_recall_good_medium_bad") or [None, None, None]
        row.update(
            {
                "status": data.get("status"),
                "split_counts": data.get("split_counts"),
                "best_epoch": data.get("best_epoch"),
                "test_acc": report.get("final_acc"),
                "base_acc": report.get("base_acc"),
                "good_recall": recalls[0],
                "medium_recall": recalls[1],
                "bad_recall": recalls[2],
                "denoise_score": metrics.get("denoise_score"),
                "snr_gain": metrics.get("snr_improve_db_mean"),
                "mse_ratio": metrics.get("mse_ratio_pred_over_noisy"),
                "visual_review": str(out_dir / "visual_review.md"),
                "balanced_gallery": str(out_dir / "debug" / "balanced_gallery.png"),
                "hard_bad_gallery": str(out_dir / "debug" / "hard_bad_gallery.png"),
            }
        )
    return row


def summarize_pure_run(spec: dict[str, Any]) -> dict[str, Any]:
    out_root = Path(spec["command"][spec["command"].index("--out_root") + 1])
    group = spec["command"][spec["command"].index("--group") + 1]
    recipe = spec["command"][spec["command"].index("--recipe") + 1]
    report_path = out_root / group / recipe / "test_report.json"
    row = {
        "stage": spec["stage"],
        "kind": spec["kind"],
        "run_id": spec["run_id"],
        "seed": spec["seed"],
        "variant": spec["variant"],
        "init_mode": spec["init_mode"],
        "output_dir": str(report_path.parent),
    }
    if report_path.exists():
        data = read_json(report_path)
        test = data.get("test", {})
        row.update(
            {
                "status": "completed",
                "split_counts": data.get("split_info"),
                "best_epoch": data.get("best_epoch"),
                "test_acc": test.get("acc"),
                "good_recall": (test.get("recall") or [None, None, None])[0],
                "medium_recall": (test.get("recall") or [None, None, None])[1],
                "bad_recall": (test.get("recall") or [None, None, None])[2],
                "test_report": str(report_path),
            }
        )
    return row


def output_dir_from_spec(spec: dict[str, Any]) -> Path:
    if spec["kind"] == "encoder_audit":
        return Path(spec["command"][spec["command"].index("--output_dir") + 1])
    out_root = Path(spec["command"][spec["command"].index("--out_root") + 1])
    group = spec["command"][spec["command"].index("--group") + 1]
    recipe = spec["command"][spec["command"].index("--recipe") + 1]
    return out_root / group / recipe


def run_spec(spec: dict[str, Any]) -> dict[str, Any]:
    run_id = spec["run_id"]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout = LOG_DIR / f"{run_id}.stdout.txt"
    stderr = LOG_DIR / f"{run_id}.stderr.txt"
    out_dir = output_dir_from_spec(spec)
    if spec["kind"] == "encoder_audit" and (out_dir / "run_summary.json").exists():
        row = summarize_encoder_run(spec, out_dir)
        row.update({"returncode": 0, "elapsed_s": 0.0, "stdout": str(stdout), "stderr": str(stderr), "skipped_existing": True})
        return row
    if spec["kind"] == "pure_transformer" and (out_dir / "test_report.json").exists():
        row = summarize_pure_run(spec)
        row.update({"returncode": 0, "elapsed_s": 0.0, "stdout": str(stdout), "stderr": str(stderr), "skipped_existing": True})
        return row
    started = time.time()
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        proc = subprocess.run(spec["command"], cwd=str(ROOT), stdout=out, stderr=err, text=True)
    row = summarize_encoder_run(spec, out_dir) if spec["kind"] == "encoder_audit" else summarize_pure_run(spec)
    row.update({"returncode": proc.returncode, "elapsed_s": time.time() - started, "stdout": str(stdout), "stderr": str(stderr)})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E3.11f Transformer re-entry audit experiments.")
    parser.add_argument("--max_runs", type=int, default=0)
    parser.add_argument("--stage", default="all")
    args = parser.parse_args()

    specs = make_specs()
    if args.stage != "all":
        specs = [s for s in specs if s["stage"] == args.stage]
    if args.max_runs > 0:
        specs = specs[: int(args.max_runs)]

    SPECS.write_text("", encoding="utf-8")
    for spec in specs:
        append_jsonl(SPECS, {k: v for k, v in spec.items() if k != "command"} | {"command": spec["command"]})

    completed: list[dict[str, Any]] = []
    write_json(STATE, {"status": "running", "total": len(specs), "completed": 0, "current": None, "artifact": str(ARTIFACT)})
    for i, spec in enumerate(specs, start=1):
        write_json(STATE, {"status": "running", "total": len(specs), "completed": len(completed), "current": spec["run_id"], "artifact": str(ARTIFACT)})
        row = run_spec(spec)
        completed.append(row)
        append_jsonl(SUMMARY, row)
        write_json(STATE, {"status": "running", "total": len(specs), "completed": len(completed), "current": None, "last": row, "artifact": str(ARTIFACT)})
        print(json.dumps({"progress": f"{i}/{len(specs)}", **row}, ensure_ascii=False)[:2400], flush=True)
        if int(row.get("returncode", 1)) != 0:
            write_json(STATE, {"status": "failed", "total": len(specs), "completed": len(completed), "failed": row, "artifact": str(ARTIFACT)})
            raise SystemExit(int(row.get("returncode", 1)))
    write_json(STATE, {"status": "completed", "total": len(specs), "completed": len(completed), "artifact": str(ARTIFACT)})


if __name__ == "__main__":
    main()
