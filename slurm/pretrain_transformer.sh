#!/bin/bash
#SBATCH -J ptbxl_pretrain
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=NONE

set -euo pipefail

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
    raise SystemExit("CUDA is required for transformer pretraining; refusing to run on CPU.")
PY

EXPERIMENT="${EXPERIMENT:?set EXPERIMENT, e.g. e7_masked_denoise}"
ARTIFACT_DIR="${ARTIFACT_DIR:-outputs/transformer_e6_local_counterfactual}"
OUT_DIR="${OUT_DIR:-$ARTIFACT_DIR/pretrain/$EXPERIMENT}"

cmd=(
  python -u -m src.transformer_pipeline.experiments.pretrain_masked_denoise
  --force
  --artifact_dir "$ARTIFACT_DIR"
  --out_dir "$OUT_DIR"
)

[[ -n "${INIT_CHECKPOINT:-}" ]] && cmd+=(--init_checkpoint "$INIT_CHECKPOINT")
[[ -n "${SEED:-}" ]] && cmd+=(--seed "$SEED")
[[ -n "${EPOCHS:-}" ]] && cmd+=(--epochs "$EPOCHS")
[[ -n "${BATCH_SIZE:-}" ]] && cmd+=(--batch_size "$BATCH_SIZE")
[[ -n "${NUM_WORKERS:-}" ]] && cmd+=(--num_workers "$NUM_WORKERS")
[[ -n "${LR:-}" ]] && cmd+=(--lr "$LR")
[[ -n "${WEIGHT_DECAY:-}" ]] && cmd+=(--weight_decay "$WEIGHT_DECAY")
[[ -n "${DROPOUT:-}" ]] && cmd+=(--dropout "$DROPOUT")
[[ -n "${MASK_RATIO:-}" ]] && cmd+=(--mask_ratio "$MASK_RATIO")
[[ -n "${LAMBDA_MASK:-}" ]] && cmd+=(--lambda_mask "$LAMBDA_MASK")
[[ -n "${LAMBDA_FULL:-}" ]] && cmd+=(--lambda_full "$LAMBDA_FULL")

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
