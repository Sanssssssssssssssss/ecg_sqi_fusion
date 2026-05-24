#!/bin/bash
#SBATCH -J e311diag_data
#SBATCH -A mphil-dis-sl2-cpu
#SBATCH -p cclake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-5%3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=NONE

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl
module load python/3.11.0-icl

cd /home/cx272/final_project/ecg_sqi_fusion
source .venv/bin/activate
mkdir -p logs reports

ROOT_OUT="outputs/transformer_e311_margin_snr_sweep"
SOURCE_ARTIFACT_DIR="${SOURCE_ARTIFACT_DIR:-outputs/transformer}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    VARIANT="e311b_snr_gap_e310_morph"
    TITLE="E3.11b wide-SNR + E3.10 morphology"
    ;;
  1)
    VARIANT="e311c_snr_gap_relaxed_morph"
    TITLE="E3.11c wide-SNR + relaxed morphology"
    ;;
  2)
    VARIANT="e311d_snr_primary_good_guard"
    TITLE="E3.11d wide-SNR primary + good guard"
    ;;
  3)
    VARIANT="e311e_snr_only_visual"
    TITLE="E3.11e wide-SNR only visual"
    ;;
  4)
    VARIANT="e311f_lite_e310_morph"
    TITLE="E3.11f lite-SNR + E3.10 morphology"
    ;;
  5)
    VARIANT="e311g_lite_snr_primary"
    TITLE="E3.11g lite-SNR primary"
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

ARTIFACT_DIR="${ROOT_OUT}/${VARIANT}"
SQI_OUT_DIR="${ARTIFACT_DIR}/sqi_ml_three_class"

echo "VARIANT=$VARIANT"
echo "ARTIFACT_DIR=$ARTIFACT_DIR"
echo "SOURCE_ARTIFACT_DIR=$SOURCE_ARTIFACT_DIR"

python -u -m src.transformer_pipeline.noise.synthesize_morph_damage_triplet \
  --force \
  --verbose \
  --artifact_dir "$ARTIFACT_DIR" \
  --source_artifact_dir "$SOURCE_ARTIFACT_DIR" \
  --label_version "$VARIANT" \
  --noise_kinds em,ma,mix \
  --group_retries "${GROUP_RETRIES:-64}" \
  --max_train_clean "${MAX_TRAIN_CLEAN:-4000}" \
  --max_val_clean "${MAX_VAL_CLEAN:-800}" \
  --max_test_clean "${MAX_TEST_CLEAN:-800}"

python -u -m src.transformer_pipeline.noise.make_rr_noise_level \
  --force \
  --verbose \
  --artifact_dir "$ARTIFACT_DIR"

python -u -m src.transformer_pipeline.diagnostics.viz_morph_triplet_samples \
  --artifact_dir "$ARTIFACT_DIR" \
  --prefix "$VARIANT" \
  --title "$TITLE" \
  --split test \
  --triplets 8 \
  --examples_per_cell 2

python -u -m src.transformer_pipeline.sqi_ml_multiclass \
  --force \
  --verbose \
  --transformer_artifact_dir "$ARTIFACT_DIR" \
  --out_dir "$SQI_OUT_DIR"

python -u -m src.transformer_pipeline.diagnostics.summarize_e311_margin_snr_sweep
