#!/bin/bash
# E3.11f round-4 high-LR basin sweep.
# Round 3 found the best single model at lr=6.25e-5 but the most stable basin
# around lr=5.75e-5. Dropout=0.075 did not beat dropout=0.10, so this round
# keeps the simple recipe and probes only LR/seed in the upper basin.
#SBATCH -J e311_r4
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
    raise SystemExit("CUDA is required for E3.11 round-4 grid; refusing to run on CPU.")
PY

ROOT_OUT="${ROOT_OUT:-outputs/transformer_e311_mainline_strict}"
VARIANT="${VARIANT:-e311f_lite_e310_morph}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/transformer_e39a_smooth_morph_triplet/models/e39_d1_smooth_morph_cls_raw/ckpt_best_val.pt}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Columns:
# run|lr|seed
specs=(
  "r4_lr575_seed4|5.75e-5|4"
  "r4_lr575_seed5|5.75e-5|5"
  "r4_lr575_seed6|5.75e-5|6"
  "r4_lr575_seed7|5.75e-5|7"
  "r4_lr61_seed0|6.1e-5|0"
  "r4_lr61_seed1|6.1e-5|1"
  "r4_lr61_seed2|6.1e-5|2"
  "r4_lr61_seed3|6.1e-5|3"
  "r4_lr625_seed4|6.25e-5|4"
  "r4_lr625_seed5|6.25e-5|5"
  "r4_lr625_seed6|6.25e-5|6"
  "r4_lr625_seed7|6.25e-5|7"
  "r4_lr64_seed0|6.4e-5|0"
  "r4_lr64_seed1|6.4e-5|1"
  "r4_lr64_seed2|6.4e-5|2"
  "r4_lr64_seed3|6.4e-5|3"
  "r4_lr65_seed0|6.5e-5|0"
  "r4_lr65_seed1|6.5e-5|1"
  "r4_lr65_seed2|6.5e-5|2"
  "r4_lr65_seed3|6.5e-5|3"
  "r4_lr675_seed0|6.75e-5|0"
  "r4_lr675_seed1|6.75e-5|1"
  "r4_lr675_seed2|6.75e-5|2"
  "r4_lr675_seed3|6.75e-5|3"
)

if (( TASK_ID < 0 || TASK_ID >= ${#specs[@]} )); then
  echo "Bad TASK_ID=$TASK_ID; valid 0..$((${#specs[@]} - 1))"
  exit 2
fi

IFS='|' read -r run lr seed <<< "${specs[$TASK_ID]}"

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
  --epochs 22
  --batch_size 32
  --lr "$lr"
  --lr_eta_min 4e-6
  --weight_decay 0.03
  --dropout 0.10
  --cls_pool cls
  --input_mode raw
  --snr_head
  --lambda_snr 0.05
  --lambda_rank 0
  --label_smoothing 0
  --class_weight_good 1
  --class_weight_medium 1
  --class_weight_bad 1
  --init_checkpoint "$INIT_CHECKPOINT"
  --select_best_by val_acc
  --use_positional_embedding
  --e_cls 22
  --e_denoise 0
  --e_level 0
  --lambda_den 0
  --lambda_lvl 0
)

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

python -m src.transformer_pipeline.analyze_training \
  --model_dir "$artifact_dir/models/$experiment_name"

python -u -m src.transformer_pipeline.diagnostics.summarize_e311_mainline_grid \
  --root_out "$ROOT_OUT"
