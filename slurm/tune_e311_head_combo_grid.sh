#!/bin/bash
# E3.11f head-combination grid.
# This sweep keeps the current best data/model base and varies only auxiliary
# heads/losses that are already implemented in train.py.
#SBATCH -J e311_heads
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-59%8
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
    raise SystemExit("CUDA is required for E3.11 head-combo grid; refusing to run on CPU.")
PY

ROOT_OUT="${ROOT_OUT:-outputs/transformer_e311_mainline_strict}"
VARIANT="${VARIANT:-e311f_lite_e310_morph}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/transformer_e39a_smooth_morph_triplet/models/e39_d1_smooth_morph_cls_raw/ckpt_best_val.pt}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Columns:
# run|lr|dropout|epochs|seed|snr_l|ord_l|noise_l|mask_l|rank_l|rank_margin|aux|den_l|lvl_l|bad_den_w|e_uncert|good_w|medium_w|bad_w|label_smoothing|weight_decay
specs=(
  "hc00_no_snr_lr575_s1|5.75e-5|0.10|22|1|0|0|0|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc01_base_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc02_base_lr575_s6|5.75e-5|0.10|22|6|0.05|0|0|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc03_base_lr625_s1|6.25e-5|0.10|22|1|0.05|0|0|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc04_base_lr625_s6|6.25e-5|0.10|22|6|0.05|0|0|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc05_snr003_lr575_s1|5.75e-5|0.10|22|1|0.03|0|0|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc06_snr0075_lr575_s1|5.75e-5|0.10|22|1|0.075|0|0|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc07_snr010_lr575_s1|5.75e-5|0.10|22|1|0.10|0|0|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"

  "hc08_ord003_lr575_s1|5.75e-5|0.10|22|1|0.05|0.03|0|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc09_ord005_lr575_s1|5.75e-5|0.10|22|1|0.05|0.05|0|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc10_ord010_lr575_s1|5.75e-5|0.10|22|1|0.05|0.10|0|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc11_ord005_lr625_s1|6.25e-5|0.10|22|1|0.05|0.05|0|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc12_ord005_lr575_s6|5.75e-5|0.10|22|6|0.05|0.05|0|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"

  "hc13_noise002_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0.02|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc14_noise005_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0.05|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc15_noise010_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0.10|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc16_noise002_lr625_s1|6.25e-5|0.10|22|1|0.05|0|0.02|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc17_noise002_lr575_s6|5.75e-5|0.10|22|6|0.05|0|0.02|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"

  "hc18_mask005_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0|0.005|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc19_mask010_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0|0.01|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc20_mask020_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0|0.02|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc21_mask050_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0|0.05|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc22_mask010_lr625_s1|6.25e-5|0.10|22|1|0.05|0|0|0.01|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc23_mask010_lr575_s6|5.75e-5|0.10|22|6|0.05|0|0|0.01|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"

  "hc24_rank010_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0|0|0.01|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc25_rank020_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0|0|0.02|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc26_rank010_lr625_s1|6.25e-5|0.10|22|1|0.05|0|0|0|0.01|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc27_snr003_rank010_lr575_s1|5.75e-5|0.10|22|1|0.03|0|0|0|0.01|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"

  "hc28_ord005_noise002_lr575_s1|5.75e-5|0.10|22|1|0.05|0.05|0.02|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc29_ord005_noise005_lr575_s1|5.75e-5|0.10|22|1|0.05|0.05|0.05|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc30_ord003_noise002_lr575_s1|5.75e-5|0.10|22|1|0.05|0.03|0.02|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc31_ord005_mask010_lr575_s1|5.75e-5|0.10|22|1|0.05|0.05|0|0.01|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc32_noise002_mask010_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0.02|0.01|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc33_ord005_noise002_mask010_lr575_s1|5.75e-5|0.10|22|1|0.05|0.05|0.02|0.01|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc34_ord005_rank010_lr575_s1|5.75e-5|0.10|22|1|0.05|0.05|0|0|0.01|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc35_noise002_rank010_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0.02|0|0.01|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc36_mask010_rank010_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0|0.01|0.01|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc37_ord005_noise002_rank010_lr575_s1|5.75e-5|0.10|22|1|0.05|0.05|0.02|0|0.01|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc38_ord003_mask005_rank010_lr575_s1|5.75e-5|0.10|22|1|0.05|0.03|0|0.005|0.01|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc39_ord003_noise002_mask005_rank010_lr575_s1|5.75e-5|0.10|22|1|0.05|0.03|0.02|0.005|0.01|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"

  "hc40_levelonly_l1_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0|0|0|0.10|level|0|1|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc41_levelonly_l05_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0|0|0|0.10|level|0|0.5|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc42_den20_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0|0|0|0.10|den|20|0|0.25|0|1.00|1.00|1.00|0.000|0.03"
  "hc43_den40_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0|0|0|0.10|den|40|0|0.25|0|1.00|1.00|1.00|0.000|0.03"
  "hc44_den20_lvl1_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0|0|0|0.10|den_level|20|1|0.25|0|1.00|1.00|1.00|0.000|0.03"
  "hc45_den40_lvl1_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0|0|0|0.10|den_level|40|1|0.25|0|1.00|1.00|1.00|0.000|0.03"
  "hc46_den20_ord005_lr575_s1|5.75e-5|0.10|22|1|0.05|0.05|0|0|0|0.10|den|20|0|0.25|0|1.00|1.00|1.00|0.000|0.03"
  "hc47_den20_noise002_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0.02|0|0|0.10|den|20|0|0.25|0|1.00|1.00|1.00|0.000|0.03"
  "hc48_den20_mask005_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0|0.005|0|0.10|den|20|0|0.25|0|1.00|1.00|1.00|0.000|0.03"
  "hc49_den20_rank010_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0|0|0.01|0.10|den|20|0|0.25|0|1.00|1.00|1.00|0.000|0.03"
  "hc50_den20_uncert_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0|0|0|0.10|den_level_uncert|20|1|0.25|4|1.00|1.00|1.00|0.000|0.03"

  "hc51_good102_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0|0|0|0.10|none|0|0|0|0|1.02|1.00|1.00|0.000|0.03"
  "hc52_good104_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0|0|0|0.10|none|0|0|0|0|1.04|1.00|1.00|0.000|0.03"
  "hc53_medium102_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0|0|0|0.10|none|0|0|0|0|1.00|1.02|1.00|0.000|0.03"
  "hc54_ls002_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.002|0.03"
  "hc55_wd002_lr575_s1|5.75e-5|0.10|22|1|0.05|0|0|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.02"

  "hc56_ord005_noise002_lr575_s6|5.75e-5|0.10|22|6|0.05|0.05|0.02|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc57_mask005_lr575_s6|5.75e-5|0.10|22|6|0.05|0|0|0.005|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
  "hc58_den20_ord005_lr575_s6|5.75e-5|0.10|22|6|0.05|0.05|0|0|0|0.10|den|20|0|0.25|0|1.00|1.00|1.00|0.000|0.03"
  "hc59_snr0075_lr575_s6|5.75e-5|0.10|22|6|0.075|0|0|0|0|0.10|none|0|0|0|0|1.00|1.00|1.00|0.000|0.03"
)

if (( TASK_ID < 0 || TASK_ID >= ${#specs[@]} )); then
  echo "Bad TASK_ID=$TASK_ID; valid 0..$((${#specs[@]} - 1))"
  exit 2
fi

IFS='|' read -r run lr dropout epochs seed snr_l ord_l noise_l mask_l rank_l rank_margin aux den_l lvl_l bad_den_w e_uncert good_w medium_w bad_w label_smoothing weight_decay <<< "${specs[$TASK_ID]}"

artifact_dir="$ROOT_OUT/$VARIANT"
experiment_name="${VARIANT}_${run}"

if [[ ! -f "$artifact_dir/datasets/synth_10s_125hz_labels_with_level.csv" ]]; then
  echo "Missing dataset for $VARIANT at $artifact_dir. Run slurm/run_e311_mainline_data_audit.sh first."
  exit 3
fi
if [[ ! -f "$INIT_CHECKPOINT" ]]; then
  echo "Missing init checkpoint: $INIT_CHECKPOINT"
  exit 4
fi
if [[ "$mask_l" != "0" && ! -f "$artifact_dir/datasets/synth_10s_125hz_local_mask.npz" ]]; then
  echo "Missing local mask for local_mask_head at $artifact_dir"
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
  --epochs "$epochs"
  --batch_size 32
  --lr "$lr"
  --lr_eta_min 4e-6
  --weight_decay "$weight_decay"
  --dropout "$dropout"
  --cls_pool cls
  --input_mode raw
  --lambda_rank "$rank_l"
  --rank_margin "$rank_margin"
  --label_smoothing "$label_smoothing"
  --class_weight_good "$good_w"
  --class_weight_medium "$medium_w"
  --class_weight_bad "$bad_w"
  --init_checkpoint "$INIT_CHECKPOINT"
  --select_best_by val_acc
  --use_positional_embedding
)

if [[ "$snr_l" != "0" ]]; then
  cmd+=(--snr_head --lambda_snr "$snr_l")
else
  cmd+=(--lambda_snr 0)
fi
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
if [[ "$mask_l" != "0" ]]; then
  cmd+=(--local_mask_head --lambda_local_mask "$mask_l")
else
  cmd+=(--lambda_local_mask 0)
fi

case "$aux" in
  none)
    cmd+=(--e_cls "$epochs" --e_denoise 0 --e_level 0 --e_uncert 0 --lambda_den 0 --lambda_lvl 0)
    ;;
  level)
    cmd+=(--e_cls 6 --e_denoise 0 --e_level "$((epochs - 6))" --e_uncert 0 --lambda_den 0 --lambda_lvl "$lvl_l")
    ;;
  den)
    cmd+=(--e_cls 6 --e_denoise "$((epochs - 6))" --e_level 0 --e_uncert 0 --lambda_den "$den_l" --lambda_lvl 0 --bad_den_w_max "$bad_den_w" --bad_den_w_warmup_epochs 10)
    ;;
  den_level)
    cmd+=(--e_cls 6 --e_denoise 10 --e_level "$((epochs - 16))" --e_uncert 0 --lambda_den "$den_l" --lambda_lvl "$lvl_l" --bad_den_w_max "$bad_den_w" --bad_den_w_warmup_epochs 10)
    ;;
  den_level_uncert)
    cmd+=(--e_cls 6 --e_denoise 8 --e_level "$((epochs - 18))" --e_uncert "$e_uncert" --lambda_den "$den_l" --lambda_lvl "$lvl_l" --bad_den_w_max "$bad_den_w" --bad_den_w_warmup_epochs 10 --uncertainty_mode kendall)
    ;;
  *)
    echo "Unknown aux mode: $aux"
    exit 6
    ;;
esac

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
