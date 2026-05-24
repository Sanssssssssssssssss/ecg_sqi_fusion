#!/bin/bash
#SBATCH -J e311f_r1
#SBATCH -A mphil-dis-sl2-gpu
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-3%2
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
D1_CKPT="outputs/transformer_e39a_smooth_morph_triplet/models/e39_d1_smooth_morph_cls_raw/ckpt_best_val.pt"

if [ ! -f "$D1_CKPT" ]; then
  echo "Missing D1 warm-start checkpoint: $D1_CKPT" >&2
  exit 2
fi
if [ ! -f "$ARTIFACT_DIR/datasets/synth_10s_125hz_labels_with_level.csv" ]; then
  echo "Missing dataset labels: $ARTIFACT_DIR/datasets/synth_10s_125hz_labels_with_level.csv" >&2
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    RUN_TAG="r1_cls_only_snr005"
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --e_cls 24 --e_denoise 0 --e_level 0)
    ;;
  1)
    RUN_TAG="r1_cls_only_snr010"
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.10 --e_cls 24 --e_denoise 0 --e_level 0)
    ;;
  2)
    RUN_TAG="r1_delay_denoise_light"
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --e_cls 6 --e_denoise 12 --e_level 6 --lambda_den 40 --lambda_lvl 0.5)
    ;;
  3)
    RUN_TAG="r1_noise_type_aux"
    EXTRA_ARGS=(--epochs 24 --lr 3e-5 --lr_eta_min 2e-6 --weight_decay 0.03 --dropout 0.10 --snr_head --lambda_snr 0.05 --noise_type_head --lambda_noise_type 0.05)
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
  --init_checkpoint "$D1_CKPT" \
  --cls_pool cls \
  --input_mode raw \
  --lambda_rank 0 \
  --select_best_by val_acc \
  "${EXTRA_ARGS[@]}"

python -u -m src.transformer_pipeline.diagnostics.summarize_e311f_tuning
