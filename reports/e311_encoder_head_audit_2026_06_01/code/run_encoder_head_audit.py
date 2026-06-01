from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


THIS_FILE = Path(__file__).absolute()
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else next(p for p in THIS_FILE.parents if (p / "src").exists())
EXP = ROOT / "outputs" / "experiment" / "e311_sqi_denoise_classifier_grid"
SCRIPT = EXP / "encoder_head_audit.py"
ARTIFACT = EXP / "artifact" / "encoder_head_audit"
LOG_DIR = EXP / "logs" / "encoder_head_audit"
STATE = EXP / "encoder_head_audit_state.json"
SUMMARY = EXP / "encoder_head_audit_summary.jsonl"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def write_state(payload: dict) -> None:
    STATE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_summary(payload: dict) -> None:
    with SUMMARY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> int:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        {
            "run_id": "encoder_head_only_e8_seed0",
            "args": [
                "--variant", "encoder_head_only",
                "--output_dir", str(ARTIFACT / "encoder_head_only_e8_seed0"),
                "--epochs", "8",
                "--batch_size", "48",
                "--lambda_cls", "18",
                "--lambda_den", "10",
                "--class_weight", "1,1.40,1.70",
                "--lr_denoiser", "4e-5",
                "--lr_head", "6e-4",
                "--seed", "0",
            ],
        },
        {
            "run_id": "dual_branch_encoder_delta_e8_seed0",
            "args": [
                "--variant", "dual_branch",
                "--gated_delta",
                "--output_dir", str(ARTIFACT / "dual_branch_encoder_delta_e8_seed0"),
                "--epochs", "8",
                "--batch_size", "48",
                "--lambda_cls", "18",
                "--lambda_den", "10",
                "--lambda_snr", "0.03",
                "--lambda_local", "0.005",
                "--delta_scale", "0.04",
                "--class_weight", "1,1.40,1.70",
                "--lr_denoiser", "4e-5",
                "--lr_classifier", "4e-6",
                "--lr_head", "6e-4",
                "--seed", "0",
            ],
        },
    ]
    write_state({"status": "running", "total": len(specs), "completed": 0, "current": None, "specs": specs})
    completed = 0
    for spec in specs:
        run_id = spec["run_id"]
        out_dir = ARTIFACT / run_id
        stdout_path = LOG_DIR / f"{run_id}.stdout.txt"
        stderr_path = LOG_DIR / f"{run_id}.stderr.txt"
        cmd = [str(PYTHON), str(SCRIPT), *spec["args"]]
        started = time.time()
        write_state({"status": "running", "total": len(specs), "completed": completed, "current": run_id, "cmd": cmd})
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=out, stderr=err, text=True)
        elapsed = time.time() - started
        row = {
            "run_id": run_id,
            "returncode": proc.returncode,
            "elapsed_s": elapsed,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "output_dir": str(out_dir),
        }
        summary_path = out_dir / "run_summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                report = summary.get("test_report", {})
                metrics = summary.get("denoise_metrics", {})
                row.update(
                    {
                        "status": summary.get("status"),
                        "best_epoch": summary.get("best_epoch"),
                        "test_acc": report.get("final_acc"),
                        "base_acc": report.get("base_acc"),
                        "recall_good_medium_bad": report.get("final_recall_good_medium_bad"),
                        "denoise_score": metrics.get("denoise_score"),
                        "mse_ratio": metrics.get("mse_ratio_pred_over_noisy"),
                        "snr_gain": metrics.get("snr_improve_db_mean"),
                    }
                )
            except Exception as exc:
                row["summary_error"] = repr(exc)
        append_summary(row)
        if proc.returncode != 0:
            write_state({"status": "failed", "total": len(specs), "completed": completed, "failed": run_id, "row": row})
            return proc.returncode
        completed += 1
        write_state({"status": "running", "total": len(specs), "completed": completed, "current": None, "last": row})
    write_state({"status": "completed", "total": len(specs), "completed": completed, "current": None})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
