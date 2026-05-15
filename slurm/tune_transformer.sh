#!/bin/bash
#SBATCH -J ptbxl_tune
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
    raise SystemExit("CUDA is required for transformer tuning; refusing to run on CPU.")
PY

EXPERIMENT_NAME="${EXPERIMENT_NAME:?set EXPERIMENT_NAME, e.g. tune_dropout005_clsheavy}"
ARTIFACT_DIR="${ARTIFACT_DIR:-outputs/transformer}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SELECT_BEST_BY="${SELECT_BEST_BY:-val_acc}"
STAGE="${STAGE:-model}"
ONLY="${ONLY:-}"

cmd=(
  python -u -m src.transformer_pipeline.run_transformer_all
  --force
  --verbose
  --stage "$STAGE"
  --artifact_dir "$ARTIFACT_DIR"
  --seed "$SEED"
  --experiment_name "$EXPERIMENT_NAME"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --select_best_by "$SELECT_BEST_BY"
)

[[ -n "$ONLY" ]] && cmd+=(--only "$ONLY")

[[ -n "${DROPOUT:-}" ]] && cmd+=(--dropout "$DROPOUT")
[[ -n "${SOURCE_ARTIFACT_DIR:-}" ]] && cmd+=(--source_artifact_dir "$SOURCE_ARTIFACT_DIR")
[[ -n "${PRESERVE_EVAL_FROM:-}" ]] && cmd+=(--preserve_eval_from "$PRESERVE_EVAL_FROM")
[[ -n "${TRAIN_AUG_MODE:-}" ]] && cmd+=(--train_aug_mode "$TRAIN_AUG_MODE")
[[ -n "${TRAIN_AUG_K:-}" ]] && cmd+=(--train_aug_k "$TRAIN_AUG_K")
[[ -n "${TRAIN_NOISE_KINDS:-}" ]] && cmd+=(--train_noise_kinds "$TRAIN_NOISE_KINDS")
[[ "${STRATIFY_NOISE_SNR:-0}" == "1" ]] && cmd+=(--stratify_noise_snr)
[[ -n "${LR:-}" ]] && cmd+=(--lr "$LR")
[[ -n "${LR_ETA_MIN:-}" ]] && cmd+=(--lr_eta_min "$LR_ETA_MIN")
[[ -n "${WEIGHT_DECAY:-}" ]] && cmd+=(--weight_decay "$WEIGHT_DECAY")
[[ -n "${CLS_POOL:-}" ]] && cmd+=(--cls_pool "$CLS_POOL")
[[ -n "${INPUT_MODE:-}" ]] && cmd+=(--input_mode "$INPUT_MODE")
[[ "${ORDINAL_HEAD:-0}" == "1" ]] && cmd+=(--ordinal_head)
[[ "${SNR_HEAD:-0}" == "1" ]] && cmd+=(--snr_head)
[[ "${LOCAL_MASK_HEAD:-0}" == "1" ]] && cmd+=(--local_mask_head)
[[ "${NOISE_TYPE_HEAD:-0}" == "1" ]] && cmd+=(--noise_type_head)
[[ "${TEACHER_DISTILL:-0}" == "1" ]] && cmd+=(--teacher_distill)
[[ "${SQI_HEAD:-0}" == "1" ]] && cmd+=(--sqi_head)
[[ -n "${INIT_CHECKPOINT:-}" ]] && cmd+=(--init_checkpoint "$INIT_CHECKPOINT")
[[ -n "${TEACHER_TARGETS:-}" ]] && cmd+=(--teacher_targets "$TEACHER_TARGETS")
[[ -n "${LAMBDA_CLS:-}" ]] && cmd+=(--lambda_cls "$LAMBDA_CLS")
[[ -n "${LAMBDA_DEN:-}" ]] && cmd+=(--lambda_den "$LAMBDA_DEN")
[[ -n "${LAMBDA_LVL:-}" ]] && cmd+=(--lambda_lvl "$LAMBDA_LVL")
[[ -n "${LAMBDA_ORD:-}" ]] && cmd+=(--lambda_ord "$LAMBDA_ORD")
[[ -n "${LAMBDA_SNR:-}" ]] && cmd+=(--lambda_snr "$LAMBDA_SNR")
[[ -n "${LAMBDA_LOCAL_MASK:-}" ]] && cmd+=(--lambda_local_mask "$LAMBDA_LOCAL_MASK")
[[ -n "${LAMBDA_NOISE_TYPE:-}" ]] && cmd+=(--lambda_noise_type "$LAMBDA_NOISE_TYPE")
[[ -n "${LAMBDA_TEACHER:-}" ]] && cmd+=(--lambda_teacher "$LAMBDA_TEACHER")
[[ -n "${LAMBDA_SQI:-}" ]] && cmd+=(--lambda_sqi "$LAMBDA_SQI")
[[ -n "${LAMBDA_RANK:-}" ]] && cmd+=(--lambda_rank "$LAMBDA_RANK")
[[ -n "${RANK_MARGIN:-}" ]] && cmd+=(--rank_margin "$RANK_MARGIN")
[[ -n "${TEACHER_TEMPERATURE:-}" ]] && cmd+=(--teacher_temperature "$TEACHER_TEMPERATURE")
[[ -n "${LABEL_SMOOTHING:-}" ]] && cmd+=(--label_smoothing "$LABEL_SMOOTHING")
[[ -n "${CLASS_WEIGHT_GOOD:-}" ]] && cmd+=(--class_weight_good "$CLASS_WEIGHT_GOOD")
[[ -n "${CLASS_WEIGHT_MEDIUM:-}" ]] && cmd+=(--class_weight_medium "$CLASS_WEIGHT_MEDIUM")
[[ -n "${CLASS_WEIGHT_BAD:-}" ]] && cmd+=(--class_weight_bad "$CLASS_WEIGHT_BAD")
[[ -n "${UNCERTAINTY_MODE:-}" ]] && cmd+=(--uncertainty_mode "$UNCERTAINTY_MODE")
[[ -n "${BAD_DEN_W_MAX:-}" ]] && cmd+=(--bad_den_w_max "$BAD_DEN_W_MAX")
[[ -n "${BAD_DEN_W_WARMUP_EPOCHS:-}" ]] && cmd+=(--bad_den_w_warmup_epochs "$BAD_DEN_W_WARMUP_EPOCHS")
[[ -n "${E_CLS:-}" ]] && cmd+=(--e_cls "$E_CLS")
[[ -n "${E_DENOISE:-}" ]] && cmd+=(--e_denoise "$E_DENOISE")
[[ -n "${E_LEVEL:-}" ]] && cmd+=(--e_level "$E_LEVEL")
[[ -n "${E_UNCERT:-}" ]] && cmd+=(--e_uncert "$E_UNCERT")
[[ -n "${NUM_WORKERS:-}" ]] && cmd+=(--num_workers "$NUM_WORKERS")
[[ "${PIN_MEMORY:-0}" == "1" ]] && cmd+=(--pin_memory)
[[ "${EARLY_STOP:-0}" == "1" ]] && cmd+=(--early_stop)
[[ -n "${EARLYSTOP_PATIENCE:-}" ]] && cmd+=(--earlystop_patience "$EARLYSTOP_PATIENCE")
[[ -n "${EARLYSTOP_MIN_DELTA:-}" ]] && cmd+=(--earlystop_min_delta "$EARLYSTOP_MIN_DELTA")
[[ -n "${EARLYSTOP_START_EPOCH:-}" ]] && cmd+=(--earlystop_start_epoch "$EARLYSTOP_START_EPOCH")

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

python -m src.transformer_pipeline.analyze_training \
  --model_dir "$ARTIFACT_DIR/models/$EXPERIMENT_NAME"
