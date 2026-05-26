#!/bin/bash
# Generate the E3.11 mainline data variants and their CPU-side audits.
# Can be run either via sbatch or directly from the repo root.
#SBATCH -J e311_data
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p cclake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=NONE

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  . /etc/profile.d/modules.sh
  module purge
  module load rhel8/default-icl
  module load python/3.11.0-icl
fi

cd /home/cx272/final_project/ecg_sqi_fusion
source .venv/bin/activate
mkdir -p logs

ROOT_OUT="${ROOT_OUT:-outputs/transformer_e311_mainline_strict}"
SOURCE_ARTIFACT_DIR="${SOURCE_ARTIFACT_DIR:-outputs/transformer_source_strict_clean}"
FORCE_DATA="${FORCE_DATA:-0}"
GROUP_RETRIES="${GROUP_RETRIES:-24}"
MAX_TRAIN_CLEAN="${MAX_TRAIN_CLEAN:-4000}"
MAX_VAL_CLEAN="${MAX_VAL_CLEAN:-800}"
MAX_TEST_CLEAN="${MAX_TEST_CLEAN:-800}"

if [[ ! -f "$SOURCE_ARTIFACT_DIR/segments/ptbxl_leadI_x_10s_125hz.npz" || ! -f "$SOURCE_ARTIFACT_DIR/splits/ptbxl_leadI_clean_10s_125hz_split.csv" ]]; then
  echo "Building source clean Lead-I segments under $SOURCE_ARTIFACT_DIR"
  python -u -m src.transformer_pipeline.run_transformer_all \
    --verbose \
    --stage preprocess \
    --only filter_lead_i,manifest,segments,split \
    --artifact_dir "$SOURCE_ARTIFACT_DIR"
fi

force_args=()
if [[ "$FORCE_DATA" == "1" ]]; then
  force_args+=(--force)
fi

variants=(
  "e311f_lite_e310_morph|e311f_lite_e310_morph|E3.11f lite SNR + E3.10 morphology"
  "e311h_lite_relaxed_morph|e311h_lite_relaxed_morph|E3.11h lite SNR + relaxed morphology"
  "e311i_wide_relaxed_morph|e311i_wide_relaxed_morph|E3.11i wide SNR + relaxed morphology"
)

for spec in "${variants[@]}"; do
  IFS='|' read -r name label_version title <<< "$spec"
  artifact_dir="$ROOT_OUT/$name"
  sqi_out_dir="$artifact_dir/sqi_ml_three_class"
  echo
  echo "=== $title ==="
  echo "artifact_dir=$artifact_dir"

  python -u -m src.transformer_pipeline.noise.synthesize_morph_damage_triplet \
    "${force_args[@]}" \
    --verbose \
    --artifact_dir "$artifact_dir" \
    --source_artifact_dir "$SOURCE_ARTIFACT_DIR" \
    --label_version "$label_version" \
    --noise_kinds em,ma,mix \
    --group_retries "$GROUP_RETRIES" \
    --max_train_clean "$MAX_TRAIN_CLEAN" \
    --max_val_clean "$MAX_VAL_CLEAN" \
    --max_test_clean "$MAX_TEST_CLEAN"

  python -u -m src.transformer_pipeline.noise.make_rr_noise_level \
    "${force_args[@]}" \
    --verbose \
    --artifact_dir "$artifact_dir"

  python -u -m src.transformer_pipeline.diagnostics.viz_morph_triplet_samples \
    --artifact_dir "$artifact_dir" \
    --prefix "$name" \
    --split test \
    --triplets 8 \
    --examples_per_cell 2

  python -u -m src.transformer_pipeline.sqi_ml_multiclass \
    "${force_args[@]}" \
    --verbose \
    --transformer_artifact_dir "$artifact_dir" \
    --out_dir "$sqi_out_dir"
done

python -u -m src.transformer_pipeline.diagnostics.summarize_e311_mainline_grid \
  --root_out "$ROOT_OUT"
