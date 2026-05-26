#!/bin/bash
# E3.11f round-2 model sweep.
# This follows the stage-1 result: E3.11f + CLS + positional embedding is the
# only branch close to 0.94, while relaxed/wide data variants are diagnostic.
# Keep the sweep local around the current best instead of expanding heads/data.
#SBATCH -J e311_r2
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --array=0-23%8
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

python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda?", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for E3.11 round-2 grid; refusing to run on CPU.")
PY

ROOT_OUT="${ROOT_OUT:-outputs/transformer_e311_mainline_strict}"
VARIANT="${VARIANT:-e311f_lite_e310_morph}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/transformer_e39a_smooth_morph_triplet/models/e39_d1_smooth_morph_cls_raw/ckpt_best_val.pt}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Columns:
# run|pool|snr_lambda|lr|dropout|input_mode|good_w|medium_w|bad_w|epochs|batch|label_smoothing|weight_decay|seed|select_best_by|noise_type
specs=(
  "r2_lr3_seed2_pos|cls|0.05|3e-5|0.10|raw|1.00|1.00|1.00|24|32|0.000|0.03|2|val_acc|0"
  "r2_lr3_seed3_pos|cls|0.05|3e-5|0.10|raw|1.00|1.00|1.00|24|32|0.000|0.03|3|val_acc|0"
  "r2_lr4_pos|cls|0.05|4e-5|0.10|raw|1.00|1.00|1.00|22|32|0.000|0.03|0|val_acc|0"
  "r2_lr45_pos|cls|0.05|4.5e-5|0.10|raw|1.00|1.00|1.00|22|32|0.000|0.03|0|val_acc|0"
  "r2_lr5_seed1_pos|cls|0.05|5e-5|0.10|raw|1.00|1.00|1.00|22|32|0.000|0.03|1|val_acc|0"
  "r2_lr5_seed2_pos|cls|0.05|5e-5|0.10|raw|1.00|1.00|1.00|22|32|0.000|0.03|2|val_acc|0"
  "r2_lr5_seed3_pos|cls|0.05|5e-5|0.10|raw|1.00|1.00|1.00|22|32|0.000|0.03|3|val_acc|0"
  "r2_lr55_pos|cls|0.05|5.5e-5|0.10|raw|1.00|1.00|1.00|22|32|0.000|0.03|0|val_acc|0"
  "r2_lr6_pos|cls|0.05|6e-5|0.10|raw|1.00|1.00|1.00|22|32|0.000|0.03|0|val_acc|0"
  "r2_lr7_pos|cls|0.05|7e-5|0.10|raw|1.00|1.00|1.00|20|32|0.000|0.03|0|val_acc|0"
  "r2_snr004_lr5_pos|cls|0.04|5e-5|0.10|raw|1.00|1.00|1.00|22|32|0.000|0.03|0|val_acc|0"
  "r2_snr006_lr5_pos|cls|0.06|5e-5|0.10|raw|1.00|1.00|1.00|22|32|0.000|0.03|0|val_acc|0"
  "r2_snr0075_lr5_pos|cls|0.075|5e-5|0.10|raw|1.00|1.00|1.00|22|32|0.000|0.03|0|val_acc|0"
  "r2_no_snr_lr5_pos|cls|0|5e-5|0.10|raw|1.00|1.00|1.00|22|32|0.000|0.03|0|val_acc|0"
  "r2_good102_lr5_pos|cls|0.05|5e-5|0.10|raw|1.02|1.00|1.00|22|32|0.000|0.03|0|val_acc|0"
  "r2_good104_lr5_pos|cls|0.05|5e-5|0.10|raw|1.04|1.00|1.00|22|32|0.000|0.03|0|val_acc|0"
  "r2_med102_lr5_pos|cls|0.05|5e-5|0.10|raw|1.00|1.02|1.00|22|32|0.000|0.03|0|val_acc|0"
  "r2_good102_med102_lr5_pos|cls|0.05|5e-5|0.10|raw|1.02|1.02|1.00|22|32|0.000|0.03|0|val_acc|0"
  "r2_clsmean_lr5_pos|cls_mean|0.05|5e-5|0.10|raw|1.00|1.00|1.00|22|32|0.000|0.03|0|val_acc|0"
  "r2_drop005_lr5_pos|cls|0.05|5e-5|0.05|raw|1.00|1.00|1.00|22|32|0.000|0.03|0|val_acc|0"
  "r2_drop0075_lr5_pos|cls|0.05|5e-5|0.075|raw|1.00|1.00|1.00|22|32|0.000|0.03|0|val_acc|0"
  "r2_valloss_lr5_pos|cls|0.05|5e-5|0.10|raw|1.00|1.00|1.00|22|32|0.000|0.03|0|val_loss|0"
  "r2_rawrobust_lr5_pos|cls|0.05|5e-5|0.10|raw_robust|1.00|1.00|1.00|22|32|0.000|0.03|0|val_acc|0"
  "r2_noise_lr5_pos|cls|0.05|5e-5|0.10|raw|1.00|1.00|1.00|22|32|0.000|0.03|0|val_acc|1"
)

if (( TASK_ID < 0 || TASK_ID >= ${#specs[@]} )); then
  echo "Bad TASK_ID=$TASK_ID; valid 0..$((${#specs[@]} - 1))"
  exit 2
fi

IFS='|' read -r run pool snr_lambda lr dropout input_mode good_w medium_w bad_w epochs batch label_smoothing weight_decay seed select_best_by noise_type <<< "${specs[$TASK_ID]}"

artifact_dir="$ROOT_OUT/$VARIANT"
experiment_name="${VARIANT}_${run}"

if [[ ! -f "$artifact_dir/datasets/synth_10s_125hz_labels_with_level.csv" ]]; then
  echo "Missing dataset for $VARIANT at $artifact_dir. Run slurm/run_e311_mainline_data_audit.sh first."
  exit 3
fi

cmd=(
  python -u -m src.transformer_pipeline.run_transformer_all
  --force
  --verbose
  --stage model
  --artifact_dir "$artifact_dir"
  --seed "$seed"
  --experiment_name "$experiment_name"
  --epochs "$epochs"
  --batch_size "$batch"
  --lr "$lr"
  --lr_eta_min 4e-6
  --weight_decay "$weight_decay"
  --dropout "$dropout"
  --cls_pool "$pool"
  --input_mode "$input_mode"
  --lambda_rank 0
  --label_smoothing "$label_smoothing"
  --class_weight_good "$good_w"
  --class_weight_medium "$medium_w"
  --class_weight_bad "$bad_w"
  --init_checkpoint "$INIT_CHECKPOINT"
  --select_best_by "$select_best_by"
  --use_positional_embedding
  --e_cls "$epochs"
  --e_denoise 0
  --e_level 0
  --lambda_den 0
  --lambda_lvl 0
)

if [[ "$snr_lambda" != "0" && "$snr_lambda" != "0.0" ]]; then
  cmd+=(--snr_head --lambda_snr "$snr_lambda")
fi
if [[ "$noise_type" == "1" ]]; then
  cmd+=(--noise_type_head --lambda_noise_type 0.05)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

python -m src.transformer_pipeline.analyze_training \
  --model_dir "$artifact_dir/models/$experiment_name"

python -u -m src.transformer_pipeline.diagnostics.summarize_e311_mainline_grid \
  --root_out "$ROOT_OUT"
