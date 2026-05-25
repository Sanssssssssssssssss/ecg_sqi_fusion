#!/bin/bash
#SBATCH -J e310_r1
#SBATCH -A mphil-dis-sl2-gpu
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --array=0-7%3
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

ARTIFACT_DIR="outputs/transformer_e310_smooth_morph_mild_snr"
D1_CKPT="outputs/transformer_e39a_smooth_morph_triplet/models/e39_d1_smooth_morph_cls_raw/ckpt_best_val.pt"
BATCH_SIZE_VALUE="${BATCH_SIZE:-32}"

if [ ! -f "$D1_CKPT" ]; then
  echo "Missing D1 warm-start checkpoint: $D1_CKPT" >&2
  exit 2
fi

BASE_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --e_cls 24 --e_denoise 0 --e_level 0)

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    RUN_TAG="r1_m2_seed1"
    SEED_VALUE=1
    EXTRA_ARGS=("${BASE_ARGS[@]}" --lambda_snr 0.05)
    ;;
  1)
    RUN_TAG="r1_m2_seed2"
    SEED_VALUE=2
    EXTRA_ARGS=("${BASE_ARGS[@]}" --lambda_snr 0.05)
    ;;
  2)
    RUN_TAG="r1_m2_seed3"
    SEED_VALUE=3
    EXTRA_ARGS=("${BASE_ARGS[@]}" --lambda_snr 0.05)
    ;;
  3)
    RUN_TAG="r1_snr002"
    SEED_VALUE=0
    EXTRA_ARGS=("${BASE_ARGS[@]}" --lambda_snr 0.02)
    ;;
  4)
    RUN_TAG="r1_snr0075"
    SEED_VALUE=0
    EXTRA_ARGS=("${BASE_ARGS[@]}" --lambda_snr 0.075)
    ;;
  5)
    RUN_TAG="r1_medium103"
    SEED_VALUE=0
    EXTRA_ARGS=("${BASE_ARGS[@]}" --lambda_snr 0.05 --class_weight_medium 1.03)
    ;;
  6)
    RUN_TAG="r1_medium105"
    SEED_VALUE=0
    EXTRA_ARGS=("${BASE_ARGS[@]}" --lambda_snr 0.05 --class_weight_medium 1.05)
    ;;
  7)
    RUN_TAG="r1_label_smoothing005"
    SEED_VALUE=0
    EXTRA_ARGS=("${BASE_ARGS[@]}" --lambda_snr 0.05 --label_smoothing 0.005)
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

EXPERIMENT_NAME="e310_${RUN_TAG}"

python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda?", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for transformer training; refusing to run on CPU.")
PY

echo "ARTIFACT_DIR=$ARTIFACT_DIR"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "SEED_VALUE=$SEED_VALUE"
echo "BATCH_SIZE_VALUE=$BATCH_SIZE_VALUE"
echo "EXTRA_ARGS=${EXTRA_ARGS[*]}"

python -u -m src.transformer_pipeline.run_transformer_all \
  --force \
  --verbose \
  --stage model \
  --artifact_dir "$ARTIFACT_DIR" \
  --seed "$SEED_VALUE" \
  --experiment_name "$EXPERIMENT_NAME" \
  --batch_size "$BATCH_SIZE_VALUE" \
  --init_checkpoint "$D1_CKPT" \
  --cls_pool cls \
  --input_mode raw \
  --lambda_rank 0 \
  --select_best_by val_acc \
  "${EXTRA_ARGS[@]}"

python -u -m src.transformer_pipeline.diagnostics.summarize_e310_visual_tuning
