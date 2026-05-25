#!/bin/bash
#SBATCH -J e311f_r3
#SBATCH -A mphil-dis-sl2-gpu
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-6%2
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

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    RUN_TAG="r3_lr25e5"
    EXTRA_ARGS=(--epochs 24 --lr 2.5e-5 --lr_eta_min 1.5e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --e_cls 24 --e_denoise 0 --e_level 0)
    ;;
  1)
    RUN_TAG="r3_lr4e5"
    EXTRA_ARGS=(--epochs 24 --lr 4e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --e_cls 24 --e_denoise 0 --e_level 0)
    ;;
  2)
    RUN_TAG="r3_drop15"
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.15 --snr_head --lambda_snr 0.05 --e_cls 24 --e_denoise 0 --e_level 0)
    ;;
  3)
    RUN_TAG="r3_wd05"
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.05 --dropout 0.10 --snr_head --lambda_snr 0.05 --e_cls 24 --e_denoise 0 --e_level 0)
    ;;
  4)
    RUN_TAG="r3_batch64"
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --e_cls 24 --e_denoise 0 --e_level 0)
    BATCH_SIZE="${BATCH_SIZE:-64}"
    ;;
  5)
    RUN_TAG="r3_snr002"
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.02 --e_cls 24 --e_denoise 0 --e_level 0)
    ;;
  6)
    RUN_TAG="r3_snr0075"
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.075 --e_cls 24 --e_denoise 0 --e_level 0)
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
echo "BATCH_SIZE=${BATCH_SIZE:-32}"
echo "EXTRA_ARGS=${EXTRA_ARGS[*]}"

python -u -m src.transformer_pipeline.run_transformer_all \
  --force \
  --verbose \
  --stage model \
  --artifact_dir "$ARTIFACT_DIR" \
  --seed "${SEED:-0}" \
  --experiment_name "$EXPERIMENT_NAME" \
  --batch_size "${BATCH_SIZE:-32}" \
  --init_checkpoint "$D1_CKPT" \
  --cls_pool cls \
  --input_mode raw \
  --lambda_rank 0 \
  --select_best_by val_acc \
  "${EXTRA_ARGS[@]}"

ROOT_OUT="$ROOT_OUT" python -u -m src.transformer_pipeline.diagnostics.summarize_e311f_tuning
