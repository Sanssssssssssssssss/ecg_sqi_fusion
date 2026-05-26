#!/bin/bash
# E3.11f local-mask focused round 2.
# Round 1 showed the first real gain from low-weight local mask:
# mask=0.01, lr=6.25e-5, seed=1 reached 0.9505. This script tests whether
# that is stable and whether nearby mask/LR/rank/ordinal settings improve it.
#SBATCH -J e311_hc2
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-39%8
#SBATCH --exclude=gpu-q-6
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
    raise SystemExit("CUDA is required for E3.11 head-combo round 2; refusing to run on CPU.")
PY

ROOT_OUT="${ROOT_OUT:-outputs/transformer_e311_mainline_strict}"
VARIANT="${VARIANT:-e311f_lite_e310_morph}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/transformer_e39a_smooth_morph_triplet/models/e39_d1_smooth_morph_cls_raw/ckpt_best_val.pt}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Columns:
# run|lr|seed|mask_l|ord_l|noise_l|rank_l|rank_margin
specs=(
  "hc2_m010_lr625_s0|6.25e-5|0|0.01|0|0|0|0.10"
  "hc2_m010_lr625_s1|6.25e-5|1|0.01|0|0|0|0.10"
  "hc2_m010_lr625_s2|6.25e-5|2|0.01|0|0|0|0.10"
  "hc2_m010_lr625_s3|6.25e-5|3|0.01|0|0|0|0.10"
  "hc2_m010_lr625_s4|6.25e-5|4|0.01|0|0|0|0.10"
  "hc2_m010_lr625_s5|6.25e-5|5|0.01|0|0|0|0.10"
  "hc2_m010_lr625_s6|6.25e-5|6|0.01|0|0|0|0.10"
  "hc2_m010_lr625_s7|6.25e-5|7|0.01|0|0|0|0.10"

  "hc2_m010_lr575_s0|5.75e-5|0|0.01|0|0|0|0.10"
  "hc2_m010_lr575_s1|5.75e-5|1|0.01|0|0|0|0.10"
  "hc2_m010_lr575_s2|5.75e-5|2|0.01|0|0|0|0.10"
  "hc2_m010_lr575_s3|5.75e-5|3|0.01|0|0|0|0.10"
  "hc2_m010_lr61_s0|6.1e-5|0|0.01|0|0|0|0.10"
  "hc2_m010_lr61_s1|6.1e-5|1|0.01|0|0|0|0.10"
  "hc2_m010_lr61_s2|6.1e-5|2|0.01|0|0|0|0.10"
  "hc2_m010_lr61_s3|6.1e-5|3|0.01|0|0|0|0.10"
  "hc2_m010_lr64_s0|6.4e-5|0|0.01|0|0|0|0.10"
  "hc2_m010_lr64_s1|6.4e-5|1|0.01|0|0|0|0.10"
  "hc2_m010_lr64_s2|6.4e-5|2|0.01|0|0|0|0.10"
  "hc2_m010_lr64_s3|6.4e-5|3|0.01|0|0|0|0.10"

  "hc2_m0075_lr625_s0|6.25e-5|0|0.0075|0|0|0|0.10"
  "hc2_m0075_lr625_s1|6.25e-5|1|0.0075|0|0|0|0.10"
  "hc2_m0075_lr625_s2|6.25e-5|2|0.0075|0|0|0|0.10"
  "hc2_m0075_lr625_s3|6.25e-5|3|0.0075|0|0|0|0.10"
  "hc2_m0125_lr625_s0|6.25e-5|0|0.0125|0|0|0|0.10"
  "hc2_m0125_lr625_s1|6.25e-5|1|0.0125|0|0|0|0.10"
  "hc2_m0125_lr625_s2|6.25e-5|2|0.0125|0|0|0|0.10"
  "hc2_m0125_lr625_s3|6.25e-5|3|0.0125|0|0|0|0.10"
  "hc2_m015_lr625_s0|6.25e-5|0|0.015|0|0|0|0.10"
  "hc2_m015_lr625_s1|6.25e-5|1|0.015|0|0|0|0.10"
  "hc2_m015_lr625_s2|6.25e-5|2|0.015|0|0|0|0.10"
  "hc2_m015_lr625_s3|6.25e-5|3|0.015|0|0|0|0.10"

  "hc2_m010_ord003_lr625_s1|6.25e-5|1|0.01|0.03|0|0|0.10"
  "hc2_m010_ord005_lr625_s1|6.25e-5|1|0.01|0.05|0|0|0.10"
  "hc2_m010_rank005_lr625_s1|6.25e-5|1|0.01|0|0|0.005|0.10"
  "hc2_m010_rank010_lr625_s1|6.25e-5|1|0.01|0|0|0.01|0.10"
  "hc2_m010_noise002_lr625_s1|6.25e-5|1|0.01|0|0.02|0|0.10"
  "hc2_m010_ord003_rank005_lr625_s1|6.25e-5|1|0.01|0.03|0|0.005|0.10"
  "hc2_m010_ord003_noise002_lr625_s1|6.25e-5|1|0.01|0.03|0.02|0|0.10"
  "hc2_m0075_rank005_lr625_s1|6.25e-5|1|0.0075|0|0|0.005|0.10"
)

if (( TASK_ID < 0 || TASK_ID >= ${#specs[@]} )); then
  echo "Bad TASK_ID=$TASK_ID; valid 0..$((${#specs[@]} - 1))"
  exit 2
fi

IFS='|' read -r run lr seed mask_l ord_l noise_l rank_l rank_margin <<< "${specs[$TASK_ID]}"

artifact_dir="$ROOT_OUT/$VARIANT"
experiment_name="${VARIANT}_${run}"

if [[ ! -f "$artifact_dir/datasets/synth_10s_125hz_labels_with_level.csv" ]]; then
  echo "Missing dataset for $VARIANT at $artifact_dir. Run slurm/run_e311_mainline_data_audit.sh first."
  exit 3
fi
if [[ ! -f "$artifact_dir/datasets/synth_10s_125hz_local_mask.npz" ]]; then
  echo "Missing local mask for local_mask_head at $artifact_dir"
  exit 4
fi
if [[ ! -f "$INIT_CHECKPOINT" ]]; then
  echo "Missing init checkpoint: $INIT_CHECKPOINT"
  exit 5
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
  --local_mask_head
  --lambda_local_mask "$mask_l"
  --lambda_rank "$rank_l"
  --rank_margin "$rank_margin"
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
  --e_uncert 0
  --lambda_den 0
  --lambda_lvl 0
)

if [[ "$ord_l" != "0" ]]; then
  cmd+=(--ordinal_head --lambda_ord "$ord_l")
else
  cmd+=(--lambda_ord 0)
fi
if [[ "$noise_l" != "0" ]]; then
  cmd+=(--noise_type_head --lambda_noise_type "$noise_l")
else
  cmd+=(--lambda_noise_type 0)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

python -m src.transformer_pipeline.analyze_training \
  --model_dir "$artifact_dir/models/$experiment_name"

python -u -m src.transformer_pipeline.diagnostics.summarize_e311_mainline_grid \
  --root_out "$ROOT_OUT"
python -u -m src.transformer_pipeline.diagnostics.summarize_e311_head_combo_grid \
  --root_out "$ROOT_OUT"
