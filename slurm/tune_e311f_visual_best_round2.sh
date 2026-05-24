#!/bin/bash
#SBATCH -J e311f_r2
#SBATCH -A mphil-dis-sl2-gpu
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-2%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=NONE

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.11.0-icl

cd /home/cx272/final_project/ecg_sqi_fusion
source .venv/bin/activate
mkdir -p logs reports

ROOT_OUT="${ROOT_OUT:-/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep}"
VARIANT="e311f_lite_e310_morph"
ARTIFACT_DIR="${ROOT_OUT}/${VARIANT}"

BEST_RUN="${BEST_RUN:-}"
if [ -z "$BEST_RUN" ]; then
  BEST_RUN="$(python - <<'PY'
from pathlib import Path
from src.transformer_pipeline.diagnostics.summarize_e311f_tuning import best_round1_run

root = Path("/rds-d6/user/cx272/hpc-work/ecg_sqi_fusion_outputs/transformer_e311_margin_snr_sweep")
run = best_round1_run(root / "e311f_lite_e310_morph" / "models")
if run is None:
    raise SystemExit("No completed round-1 run found.")
print(run)
PY
)"
fi

BEST_CKPT="${ARTIFACT_DIR}/models/${BEST_RUN}/ckpt_best_val.pt"
if [ ! -f "$BEST_CKPT" ]; then
  echo "Missing round-1 warm-start checkpoint: $BEST_CKPT" >&2
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    RUN_TAG="r2_${BEST_RUN#e311f_lite_e310_morph_}_lr2e5"
    EXTRA_ARGS=(--epochs 18 --lr 2e-5 --lr_eta_min 1e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --e_cls 18 --e_denoise 0 --e_level 0)
    ;;
  1)
    RUN_TAG="r2_${BEST_RUN#e311f_lite_e310_morph_}_ls003"
    EXTRA_ARGS=(--epochs 18 --lr 2e-5 --lr_eta_min 1e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --label_smoothing 0.03 --e_cls 18 --e_denoise 0 --e_level 0)
    ;;
  2)
    RUN_TAG="r2_${BEST_RUN#e311f_lite_e310_morph_}_gm_weight"
    EXTRA_ARGS=(--epochs 18 --lr 2e-5 --lr_eta_min 1e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --class_weight_good 1.05 --class_weight_medium 1.05 --class_weight_bad 1.0 --e_cls 18 --e_denoise 0 --e_level 0)
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

EXPERIMENT_NAME="${VARIANT}_${RUN_TAG}"

python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda?", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for transformer training; refusing to run on CPU.")
PY

echo "BEST_RUN=$BEST_RUN"
echo "BEST_CKPT=$BEST_CKPT"
echo "ARTIFACT_DIR=$ARTIFACT_DIR"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "EXTRA_ARGS=${EXTRA_ARGS[*]}"

python -u -m src.transformer_pipeline.run_transformer_all \
  --force \
  --verbose \
  --stage model \
  --artifact_dir "$ARTIFACT_DIR" \
  --seed "${SEED:-0}" \
  --experiment_name "$EXPERIMENT_NAME" \
  --batch_size "${BATCH_SIZE:-32}" \
  --init_checkpoint "$BEST_CKPT" \
  --cls_pool cls \
  --input_mode raw \
  --lambda_rank 0 \
  --select_best_by val_acc \
  "${EXTRA_ARGS[@]}"

python -u -m src.transformer_pipeline.diagnostics.summarize_e311f_tuning
