#!/bin/bash
#SBATCH -J sqi12_sweep
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=NONE

set -uo pipefail

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.11.0-icl

cd /home/cx272/final_project/ecg_sqi_fusion
source .venv/bin/activate
mkdir -p logs

python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda?", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for SQI transformer tuning; refusing to run on CPU.")
PY

SWEEP_NAME="${SWEEP_NAME:-transformer_12lead_sweep_v2}"
SWEEP_ROOT="${SWEEP_ROOT:-outputs/sqi/models/${SWEEP_NAME}_${SLURM_JOB_ID:-local}}"
mkdir -p "$SWEEP_ROOT"

COMMON_ARGS=(
  --epochs "${EPOCHS:-100}"
  --patience "${PATIENCE:-18}"
  --threshold_grid 2001
  --num_workers "${NUM_WORKERS:-0}"
  --device auto
)

# Format: run_name|extra args
# This is intentionally a broad, simple sweep rather than seed-only tuning.
RUN_SPECS=(
  "v01_baseline_retrain|--seed 0 --batch_size 64 --lr 0.001 --weight_decay 0.0001 --d_model 96 --n_heads 4 --n_layers 2 --ff_dim 192 --dropout 0.10 --pooling cls --patch_size 25 --stride 10"
  "v02_tiny_regularized|--seed 0 --batch_size 64 --lr 0.0005 --weight_decay 0.001 --d_model 64 --n_heads 4 --n_layers 1 --ff_dim 128 --dropout 0.20 --pooling cls --patch_size 25 --stride 10"
  "v03_small_mean|--seed 0 --batch_size 64 --lr 0.0005 --weight_decay 0.001 --d_model 64 --n_heads 4 --n_layers 1 --ff_dim 128 --dropout 0.20 --pooling mean --patch_size 25 --stride 10"
  "v04_wide_cls|--seed 0 --batch_size 64 --lr 0.0005 --weight_decay 0.0005 --d_model 128 --n_heads 4 --n_layers 2 --ff_dim 256 --dropout 0.15 --pooling cls --patch_size 25 --stride 10"
  "v05_deeper_cls|--seed 0 --batch_size 64 --lr 0.0005 --weight_decay 0.0005 --d_model 96 --n_heads 4 --n_layers 3 --ff_dim 192 --dropout 0.15 --pooling cls --patch_size 25 --stride 10"
  "v06_wide_deep_cls|--seed 0 --batch_size 48 --lr 0.0003 --weight_decay 0.001 --d_model 128 --n_heads 4 --n_layers 3 --ff_dim 256 --dropout 0.20 --pooling cls --patch_size 25 --stride 10"
  "v07_dense_patch|--seed 0 --batch_size 64 --lr 0.0008 --weight_decay 0.0005 --d_model 96 --n_heads 4 --n_layers 2 --ff_dim 192 --dropout 0.15 --pooling cls --patch_size 25 --stride 5"
  "v08_patch50_s10|--seed 0 --batch_size 64 --lr 0.0008 --weight_decay 0.0005 --d_model 96 --n_heads 4 --n_layers 2 --ff_dim 192 --dropout 0.15 --pooling cls --patch_size 50 --stride 10"
  "v09_patch50_s25|--seed 0 --batch_size 64 --lr 0.0008 --weight_decay 0.0005 --d_model 96 --n_heads 4 --n_layers 2 --ff_dim 192 --dropout 0.15 --pooling cls --patch_size 50 --stride 25"
  "v10_patch75_s25|--seed 0 --batch_size 64 --lr 0.0008 --weight_decay 0.0005 --d_model 96 --n_heads 4 --n_layers 2 --ff_dim 192 --dropout 0.15 --pooling cls --patch_size 75 --stride 25"
  "v11_patch100_s25|--seed 0 --batch_size 64 --lr 0.0008 --weight_decay 0.0005 --d_model 96 --n_heads 4 --n_layers 2 --ff_dim 192 --dropout 0.15 --pooling cls --patch_size 100 --stride 25"
  "v12_patch100_s50|--seed 0 --batch_size 64 --lr 0.0008 --weight_decay 0.0005 --d_model 96 --n_heads 4 --n_layers 2 --ff_dim 192 --dropout 0.15 --pooling cls --patch_size 100 --stride 50"
  "v13_patch125_s25|--seed 0 --batch_size 64 --lr 0.0005 --weight_decay 0.0005 --d_model 96 --n_heads 4 --n_layers 2 --ff_dim 192 --dropout 0.15 --pooling cls --patch_size 125 --stride 25"
  "v14_patch150_s50|--seed 0 --batch_size 64 --lr 0.0005 --weight_decay 0.0005 --d_model 96 --n_heads 4 --n_layers 2 --ff_dim 192 --dropout 0.15 --pooling cls --patch_size 150 --stride 50"
  "v15_patch100_mean|--seed 0 --batch_size 64 --lr 0.0008 --weight_decay 0.0005 --d_model 96 --n_heads 4 --n_layers 2 --ff_dim 192 --dropout 0.15 --pooling mean --patch_size 100 --stride 25"
  "v16_patch100_clsmean|--seed 0 --batch_size 64 --lr 0.0008 --weight_decay 0.0005 --d_model 96 --n_heads 4 --n_layers 2 --ff_dim 192 --dropout 0.15 --pooling cls_mean --patch_size 100 --stride 25"
  "v17_patch100_lowlr|--seed 0 --batch_size 64 --lr 0.0003 --weight_decay 0.0005 --d_model 96 --n_heads 4 --n_layers 2 --ff_dim 192 --dropout 0.15 --pooling cls --patch_size 100 --stride 25"
  "v18_patch100_highlr|--seed 0 --batch_size 64 --lr 0.0015 --weight_decay 0.0005 --d_model 96 --n_heads 4 --n_layers 2 --ff_dim 192 --dropout 0.15 --pooling cls --patch_size 100 --stride 25"
  "v19_patch100_drop05|--seed 0 --batch_size 64 --lr 0.0008 --weight_decay 0.0005 --d_model 96 --n_heads 4 --n_layers 2 --ff_dim 192 --dropout 0.05 --pooling cls --patch_size 100 --stride 25"
  "v20_patch100_drop25|--seed 0 --batch_size 64 --lr 0.0005 --weight_decay 0.001 --d_model 96 --n_heads 4 --n_layers 2 --ff_dim 192 --dropout 0.25 --pooling cls --patch_size 100 --stride 25"
  "v21_patch100_ls002|--seed 0 --batch_size 64 --lr 0.0008 --weight_decay 0.0005 --d_model 96 --n_heads 4 --n_layers 2 --ff_dim 192 --dropout 0.15 --pooling cls --patch_size 100 --stride 25 --label_smoothing 0.02"
  "v22_patch100_ls005|--seed 0 --batch_size 64 --lr 0.0008 --weight_decay 0.0005 --d_model 96 --n_heads 4 --n_layers 2 --ff_dim 192 --dropout 0.15 --pooling cls --patch_size 100 --stride 25 --label_smoothing 0.05"
  "v23_patch100_valauc|--seed 0 --batch_size 64 --lr 0.0008 --weight_decay 0.0005 --d_model 96 --n_heads 4 --n_layers 2 --ff_dim 192 --dropout 0.15 --pooling cls --patch_size 100 --stride 25 --select_best_by val_auc"
  "v24_wide_patch100_valauc_ls002|--seed 0 --batch_size 48 --lr 0.0005 --weight_decay 0.001 --d_model 128 --n_heads 4 --n_layers 2 --ff_dim 256 --dropout 0.20 --pooling cls --patch_size 100 --stride 25 --label_smoothing 0.02 --select_best_by val_auc"
)

printf 'run_name\textra_args\n' > "$SWEEP_ROOT/run_matrix.tsv"
for spec in "${RUN_SPECS[@]}"; do
  printf '%s\t%s\n' "${spec%%|*}" "${spec#*|}" >> "$SWEEP_ROOT/run_matrix.tsv"
done

echo "Sweep root: $SWEEP_ROOT"
echo "Total variants: ${#RUN_SPECS[@]}"

for spec in "${RUN_SPECS[@]}"; do
  run_name="${spec%%|*}"
  arg_string="${spec#*|}"
  out_dir="$SWEEP_ROOT/$run_name"
  mkdir -p "$out_dir"
  read -r -a extra_args <<< "$arg_string"

  cmd=(
    python -u src/sqi_pipeline/models/transformer_12lead.py
    "${COMMON_ARGS[@]}"
    --out_dir "$out_dir"
    "${extra_args[@]}"
  )

  echo "===== RUN $run_name ====="
  printf 'Running:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if "${cmd[@]}" 2>&1 | tee "$out_dir/train.log"; then
    echo "status=ok" > "$out_dir/status.txt"
  else
    echo "status=failed" > "$out_dir/status.txt"
  fi
done

python - "$SWEEP_ROOT" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

sweep_root = Path(sys.argv[1])
rows = []
for metrics_path in sorted(sweep_root.glob("*/metrics.json")):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cfg = metrics["config"]
    test = metrics["fixed_threshold_metrics"]["test"]
    val = metrics["fixed_threshold_metrics"]["val"]
    oracle = metrics["test_oracle_maxacc"]
    val_thr = metrics["test_at_val_selected_threshold"]
    rows.append({
        "run": metrics_path.parent.name,
        "seed": cfg["seed"],
        "test_acc": test["acc"],
        "test_auc": test["auc"],
        "test_se": test["se"],
        "test_sp": test["sp"],
        "test_oracle_acc": oracle["acc"],
        "test_val_threshold_acc": val_thr["acc"],
        "val_acc": val["acc"],
        "val_auc": val["auc"],
        "best_epoch": metrics["best_epoch"],
        "pooling": cfg.get("pooling", "cls"),
        "d_model": cfg["d_model"],
        "n_layers": cfg["n_layers"],
        "ff_dim": cfg["ff_dim"],
        "dropout": cfg["dropout"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "batch_size": cfg["batch_size"],
        "patch_size": cfg["patch_size"],
        "stride": cfg["stride"],
        "label_smoothing": cfg.get("label_smoothing", 0.0),
        "select_best_by": metrics.get("select_best_by", cfg.get("select_best_by", "val_acc")),
        "tn": test["confusion_matrix"]["tn"],
        "fp": test["confusion_matrix"]["fp"],
        "fn": test["confusion_matrix"]["fn"],
        "tp": test["confusion_matrix"]["tp"],
    })

rows.sort(key=lambda r: (r["test_acc"], r["test_auc"]), reverse=True)
csv_path = sweep_root / "summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["run"])
    writer.writeheader()
    writer.writerows(rows)

def fmt(x: float) -> str:
    return "nan" if isinstance(x, float) and math.isnan(x) else f"{x:.4f}"

md_lines = [
    "# SQI 12-Lead Transformer Sweep v2",
    "",
    f"Sweep root: `{sweep_root}`",
    "",
    f"Completed variants: {len(rows)}",
    "",
    "| Rank | Run | Test Acc | AUC | Se | Sp | Oracle | Val Acc/AUC | Key config |",
    "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
]
for i, r in enumerate(rows, start=1):
    key = (
        f"pool={r['pooling']}, d={r['d_model']}, L={r['n_layers']}, "
        f"p/s={r['patch_size']}/{r['stride']}, drop={r['dropout']}, "
        f"lr={r['lr']}, wd={r['weight_decay']}, ls={r['label_smoothing']}, "
        f"best={r['select_best_by']}"
    )
    md_lines.append(
        f"| {i} | `{r['run']}` | {fmt(r['test_acc'])} | {fmt(r['test_auc'])} | "
        f"{fmt(r['test_se'])} | {fmt(r['test_sp'])} | {fmt(r['test_oracle_acc'])} | "
        f"{fmt(r['val_acc'])}/{fmt(r['val_auc'])} | {key} |"
    )
md_lines.extend([
    "",
    "Generated files:",
    "",
    "```text",
    str(csv_path),
    str(sweep_root / "summary.md"),
    str(sweep_root / "run_matrix.tsv"),
    "```",
])
(sweep_root / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

print(f"Wrote {csv_path}")
print(f"Wrote {sweep_root / 'summary.md'}")
if rows:
    best = rows[0]
    print(f"Best: {best['run']} test_acc={best['test_acc']:.4f} auc={best['test_auc']:.4f}")
PY
