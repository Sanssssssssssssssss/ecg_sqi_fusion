#!/bin/bash
#SBATCH -J e311f_r4
#SBATCH -A mphil-dis-sl2-gpu
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-7%2
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
D1_CKPT="/home/cx272/final_project/ecg_sqi_fusion/outputs/transformer_e39a_smooth_morph_triplet/models/e39_d1_smooth_morph_cls_raw/ckpt_best_val.pt"
SELECT_BEST_BY="val_acc"
SEED_VALUE="${SEED:-0}"
BATCH_SIZE_VALUE="${BATCH_SIZE:-32}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    RUN_TAG="r4_seed1_best"
    SEED_VALUE=1
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --e_cls 24 --e_denoise 0 --e_level 0)
    ;;
  1)
    RUN_TAG="r4_seed2_best"
    SEED_VALUE=2
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --e_cls 24 --e_denoise 0 --e_level 0)
    ;;
  2)
    RUN_TAG="r4_seed3_best"
    SEED_VALUE=3
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --e_cls 24 --e_denoise 0 --e_level 0)
    ;;
  3)
    RUN_TAG="r4_dropout005"
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.05 --snr_head --lambda_snr 0.05 --e_cls 24 --e_denoise 0 --e_level 0)
    ;;
  4)
    RUN_TAG="r4_label_smoothing001"
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --label_smoothing 0.01 --e_cls 24 --e_denoise 0 --e_level 0)
    ;;
  5)
    RUN_TAG="r4_good_weight108"
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --class_weight_good 1.08 --class_weight_medium 1.00 --class_weight_bad 1.00 --e_cls 24 --e_denoise 0 --e_level 0)
    ;;
  6)
    RUN_TAG="r4_medium_weight108"
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --class_weight_good 1.00 --class_weight_medium 1.08 --class_weight_bad 1.00 --e_cls 24 --e_denoise 0 --e_level 0)
    ;;
  7)
    RUN_TAG="r4_select_val_loss"
    SELECT_BEST_BY="val_loss"
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --e_cls 24 --e_denoise 0 --e_level 0)
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

echo "D1_CKPT=$D1_CKPT"
echo "ARTIFACT_DIR=$ARTIFACT_DIR"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "SEED_VALUE=$SEED_VALUE"
echo "BATCH_SIZE_VALUE=$BATCH_SIZE_VALUE"
echo "SELECT_BEST_BY=$SELECT_BEST_BY"
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
  --select_best_by "$SELECT_BEST_BY" \
  "${EXTRA_ARGS[@]}"

ROOT_OUT="$ROOT_OUT" python -u -m src.transformer_pipeline.diagnostics.summarize_e311f_tuning
