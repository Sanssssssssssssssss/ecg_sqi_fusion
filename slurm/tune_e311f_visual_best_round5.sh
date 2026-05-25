#!/bin/bash
#SBATCH -J e311f_r5
#SBATCH -A mphil-dis-sl2-gpu
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
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
CLS_POOL_VALUE="cls"
INPUT_MODE_VALUE="raw"
BATCH_SIZE_VALUE="${BATCH_SIZE:-32}"
SEED_VALUE="${SEED:-0}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    RUN_TAG="r5_decoder_pool"
    CLS_POOL_VALUE="decoder"
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --e_cls 24 --e_denoise 0 --e_level 0)
    ;;
  1)
    RUN_TAG="r5_robust_input"
    INPUT_MODE_VALUE="robust"
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --e_cls 24 --e_denoise 0 --e_level 0)
    ;;
  2)
    RUN_TAG="r5_rank001"
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --lambda_rank 0.01 --rank_margin 0.08 --e_cls 24 --e_denoise 0 --e_level 0)
    ;;
  3)
    RUN_TAG="r5_rank002"
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --lambda_rank 0.02 --rank_margin 0.10 --e_cls 24 --e_denoise 0 --e_level 0)
    ;;
  4)
    RUN_TAG="r5_ordinal_snr"
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --ordinal_head --snr_head --lambda_ord 0.05 --lambda_snr 0.05 --e_cls 24 --e_denoise 0 --e_level 0)
    ;;
  5)
    RUN_TAG="r5_ordinal_rank001"
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --ordinal_head --snr_head --lambda_ord 0.05 --lambda_snr 0.05 --lambda_rank 0.01 --rank_margin 0.08 --e_cls 24 --e_denoise 0 --e_level 0)
    ;;
  6)
    RUN_TAG="r5_long_earlystop"
    EXTRA_ARGS=(--epochs 36 --lr 3e-5 --lr_eta_min 1e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --e_cls 36 --e_denoise 0 --e_level 0 --early_stop --earlystop_start_epoch 12 --earlystop_patience 8 --earlystop_min_delta 0.0005)
    ;;
  7)
    RUN_TAG="r5_tiny_denoise_curriculum"
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --e_cls 16 --e_denoise 8 --e_level 0 --lambda_den 10 --bad_den_w_max 0.10 --bad_den_w_warmup_epochs 8)
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
echo "CLS_POOL_VALUE=$CLS_POOL_VALUE"
echo "INPUT_MODE_VALUE=$INPUT_MODE_VALUE"
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
  --cls_pool "$CLS_POOL_VALUE" \
  --input_mode "$INPUT_MODE_VALUE" \
  --select_best_by val_acc \
  "${EXTRA_ARGS[@]}"

ROOT_OUT="$ROOT_OUT" python -u -m src.transformer_pipeline.diagnostics.summarize_e311f_tuning
