#!/bin/bash
#SBATCH -J e31x_snr
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
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
mkdir -p logs

D1_CKPT="outputs/transformer_e39a_smooth_morph_triplet/models/e39_d1_smooth_morph_cls_raw/ckpt_best_val.pt"
if [ ! -f "$D1_CKPT" ]; then
  echo "Missing D1 warm-start checkpoint: $D1_CKPT" >&2
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    DATA_TAG="e310"
    ARTIFACT_DIR="outputs/transformer_e310_smooth_morph_mild_snr"
    RUN_TAG="m1_d1warm_lr3e5"
    EXTRA_ARGS=(--init_checkpoint "$D1_CKPT" --epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10)
    ;;
  1)
    DATA_TAG="e310"
    ARTIFACT_DIR="outputs/transformer_e310_smooth_morph_mild_snr"
    RUN_TAG="m2_d1warm_snr005"
    EXTRA_ARGS=(--init_checkpoint "$D1_CKPT" --epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05)
    ;;
  2)
    DATA_TAG="e310"
    ARTIFACT_DIR="outputs/transformer_e310_smooth_morph_mild_snr"
    RUN_TAG="m3_d1warm_snr005_lowden"
    EXTRA_ARGS=(--init_checkpoint "$D1_CKPT" --epochs 25 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --e_cls 5 --e_denoise 15 --e_level 5 --lambda_den 60)
    ;;
  3)
    DATA_TAG="e310"
    ARTIFACT_DIR="outputs/transformer_e310_smooth_morph_mild_snr"
    RUN_TAG="m4_d1warm_snr005_lowden_ntype005"
    EXTRA_ARGS=(--init_checkpoint "$D1_CKPT" --epochs 25 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --e_cls 5 --e_denoise 15 --e_level 5 --lambda_den 60 --noise_type_head --lambda_noise_type 0.05)
    ;;
  4)
    DATA_TAG="e311"
    ARTIFACT_DIR="outputs/transformer_e311_visual_gap"
    RUN_TAG="m1_d1warm_lr3e5"
    EXTRA_ARGS=(--init_checkpoint "$D1_CKPT" --epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10)
    ;;
  5)
    DATA_TAG="e311"
    ARTIFACT_DIR="outputs/transformer_e311_visual_gap"
    RUN_TAG="m2_d1warm_snr005"
    EXTRA_ARGS=(--init_checkpoint "$D1_CKPT" --epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05)
    ;;
  6)
    DATA_TAG="e311"
    ARTIFACT_DIR="outputs/transformer_e311_visual_gap"
    RUN_TAG="m3_d1warm_snr005_lowden"
    EXTRA_ARGS=(--init_checkpoint "$D1_CKPT" --epochs 25 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --e_cls 5 --e_denoise 15 --e_level 5 --lambda_den 60)
    ;;
  7)
    DATA_TAG="e311"
    ARTIFACT_DIR="outputs/transformer_e311_visual_gap"
    RUN_TAG="m4_d1warm_snr005_lowden_ntype005"
    EXTRA_ARGS=(--init_checkpoint "$D1_CKPT" --epochs 25 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --e_cls 5 --e_denoise 15 --e_level 5 --lambda_den 60 --noise_type_head --lambda_noise_type 0.05)
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

if [ ! -f "$ARTIFACT_DIR/datasets/synth_10s_125hz_labels_with_level.csv" ]; then
  echo "Missing dataset labels: $ARTIFACT_DIR/datasets/synth_10s_125hz_labels_with_level.csv" >&2
  exit 2
fi

EXPERIMENT_NAME="${DATA_TAG}_${RUN_TAG}"

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
echo "EXTRA_ARGS=${EXTRA_ARGS[*]}"

python -u -m src.transformer_pipeline.run_transformer_all \
  --force \
  --verbose \
  --stage model \
  --artifact_dir "$ARTIFACT_DIR" \
  --seed "${SEED:-0}" \
  --experiment_name "$EXPERIMENT_NAME" \
  --batch_size "${BATCH_SIZE:-32}" \
  --cls_pool cls \
  --input_mode raw \
  --lambda_rank 0 \
  --select_best_by val_acc \
  "${EXTRA_ARGS[@]}"
