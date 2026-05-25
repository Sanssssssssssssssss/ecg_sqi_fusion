#!/bin/bash
# E3.11 mainline model/data grid.
# Stage 1 intentionally mixes a broad model grid on E3.11f with small screens
# for relaxed-morph data variants. Bad branches should be cancelled or not
# expanded after the first report.
#SBATCH -J e311_grid
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --array=0-35%8
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
    raise SystemExit("CUDA is required for E3.11 grid; refusing to run on CPU.")
PY

ROOT_OUT="${ROOT_OUT:-outputs/transformer_e311_mainline}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/transformer_e39a_smooth_morph_triplet/models/e39_d1_smooth_morph_cls_raw/ckpt_best_val.pt}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Columns:
# variant|run|pool|pos|snr_lambda|lr|dropout|ordinal|noise_type|denoise|good_w|medium_w|bad_w|epochs|batch|label_smoothing|ord_lambda|noise_lambda|weight_decay|seed
specs=(
  "e311f_lite_e310_morph|f00_anchor|cls|0|0.05|3e-5|0.10|0|0|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311f_lite_e310_morph|f01_pos|cls|1|0.05|3e-5|0.10|0|0|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311f_lite_e310_morph|f02_clsmean|cls_mean|0|0.05|3e-5|0.10|0|0|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311f_lite_e310_morph|f03_clsmean_pos|cls_mean|1|0.05|3e-5|0.10|0|0|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311f_lite_e310_morph|f04_lr2_pos|cls|1|0.05|2e-5|0.10|0|0|0|1.00|1.00|1.00|26|32|0.000|0.10|0.05|0.03|0"
  "e311f_lite_e310_morph|f05_lr5_pos|cls|1|0.05|5e-5|0.10|0|0|0|1.00|1.00|1.00|22|32|0.000|0.10|0.05|0.03|0"
  "e311f_lite_e310_morph|f06_drop015_pos|cls|1|0.05|3e-5|0.15|0|0|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311f_lite_e310_morph|f07_drop020_pos|cls|1|0.05|3e-5|0.20|0|0|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311f_lite_e310_morph|f08_snr003_pos|cls|1|0.03|3e-5|0.10|0|0|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311f_lite_e310_morph|f09_snr010_pos|cls|1|0.10|3e-5|0.10|0|0|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311f_lite_e310_morph|f10_ord_pos|cls|1|0.05|3e-5|0.10|1|0|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311f_lite_e310_morph|f11_noise_pos|cls|1|0.05|3e-5|0.10|0|1|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311f_lite_e310_morph|f12_ord_noise_pos|cls|1|0.05|3e-5|0.10|1|1|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311f_lite_e310_morph|f13_lateden_pos|cls|1|0.05|3e-5|0.10|0|0|1|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311f_lite_e310_morph|f14_lateden_clsmean_pos|cls_mean|1|0.05|3e-5|0.10|0|0|1|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311f_lite_e310_morph|f15_lateden_noise_pos|cls|1|0.05|3e-5|0.10|0|1|1|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311f_lite_e310_morph|f16_good104_pos|cls|1|0.05|3e-5|0.10|0|0|0|1.04|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311f_lite_e310_morph|f17_med104_pos|cls|1|0.05|3e-5|0.10|0|0|0|1.00|1.04|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311f_lite_e310_morph|f18_bad104_pos|cls|1|0.05|3e-5|0.10|0|0|0|1.00|1.00|1.04|24|32|0.000|0.10|0.05|0.03|0"
  "e311f_lite_e310_morph|f19_ls003_pos|cls|1|0.05|3e-5|0.10|0|0|0|1.00|1.00|1.00|24|32|0.003|0.10|0.05|0.03|0"
  "e311f_lite_e310_morph|f20_wd001_pos|cls|1|0.05|3e-5|0.10|0|0|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.01|0"
  "e311f_lite_e310_morph|f21_wd005_pos|cls|1|0.05|3e-5|0.10|0|0|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.05|0"
  "e311f_lite_e310_morph|f22_batch64_pos|cls|1|0.05|3e-5|0.10|0|0|0|1.00|1.00|1.00|24|64|0.000|0.10|0.05|0.03|0"
  "e311f_lite_e310_morph|f23_seed1_pos|cls|1|0.05|3e-5|0.10|0|0|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|1"

  "e311h_lite_relaxed_morph|h00_anchor|cls|0|0.05|3e-5|0.10|0|0|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311h_lite_relaxed_morph|h01_pos|cls|1|0.05|3e-5|0.10|0|0|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311h_lite_relaxed_morph|h02_clsmean_pos|cls_mean|1|0.05|3e-5|0.10|0|0|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311h_lite_relaxed_morph|h03_snr010_pos|cls|1|0.10|3e-5|0.10|0|0|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311h_lite_relaxed_morph|h04_ord_pos|cls|1|0.05|3e-5|0.10|1|0|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311h_lite_relaxed_morph|h05_lateden_pos|cls|1|0.05|3e-5|0.10|0|0|1|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"

  "e311i_wide_relaxed_morph|i00_anchor|cls|0|0.05|3e-5|0.10|0|0|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311i_wide_relaxed_morph|i01_pos|cls|1|0.05|3e-5|0.10|0|0|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311i_wide_relaxed_morph|i02_clsmean_pos|cls_mean|1|0.05|3e-5|0.10|0|0|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311i_wide_relaxed_morph|i03_snr010_pos|cls|1|0.10|3e-5|0.10|0|0|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311i_wide_relaxed_morph|i04_ord_pos|cls|1|0.05|3e-5|0.10|1|0|0|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
  "e311i_wide_relaxed_morph|i05_lateden_pos|cls|1|0.05|3e-5|0.10|0|0|1|1.00|1.00|1.00|24|32|0.000|0.10|0.05|0.03|0"
)

if (( TASK_ID < 0 || TASK_ID >= ${#specs[@]} )); then
  echo "Bad TASK_ID=$TASK_ID; valid 0..$((${#specs[@]} - 1))"
  exit 2
fi

IFS='|' read -r variant run pool pos snr_lambda lr dropout ordinal noise_type denoise good_w medium_w bad_w epochs batch label_smoothing ord_lambda noise_lambda weight_decay seed <<< "${specs[$TASK_ID]}"

artifact_dir="$ROOT_OUT/$variant"
experiment_name="${variant}_${run}"

if [[ ! -f "$artifact_dir/datasets/synth_10s_125hz_labels_with_level.csv" ]]; then
  echo "Missing dataset for $variant at $artifact_dir. Run slurm/run_e311_mainline_data_audit.sh first."
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
  --input_mode raw
  --snr_head
  --lambda_snr "$snr_lambda"
  --lambda_rank 0
  --label_smoothing "$label_smoothing"
  --class_weight_good "$good_w"
  --class_weight_medium "$medium_w"
  --class_weight_bad "$bad_w"
  --init_checkpoint "$INIT_CHECKPOINT"
  --select_best_by val_acc
)

if [[ "$pos" == "1" ]]; then
  cmd+=(--use_positional_embedding)
fi
if [[ "$ordinal" == "1" ]]; then
  cmd+=(--ordinal_head --lambda_ord "$ord_lambda")
fi
if [[ "$noise_type" == "1" ]]; then
  cmd+=(--noise_type_head --lambda_noise_type "$noise_lambda")
fi
if [[ "$denoise" == "1" ]]; then
  cmd+=(--e_cls 6 --e_denoise 12 --e_level 6 --lambda_den 40 --lambda_lvl 1 --bad_den_w_max 0.25 --bad_den_w_warmup_epochs 12)
else
  cmd+=(--e_cls "$epochs" --e_denoise 0 --e_level 0 --lambda_den 0 --lambda_lvl 0)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

python -m src.transformer_pipeline.analyze_training \
  --model_dir "$artifact_dir/models/$experiment_name"

python -u -m src.transformer_pipeline.diagnostics.summarize_e311_mainline_grid \
  --root_out "$ROOT_OUT"
