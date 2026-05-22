#!/bin/bash
#SBATCH -J e310_mild
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
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
    raise SystemExit("CUDA is required for E3.10 transformer training; refusing to run on CPU.")
PY

ARTIFACT_DIR="${ARTIFACT_DIR:-outputs/transformer_e310_smooth_morph_mild_snr}"
SOURCE_ARTIFACT_DIR="${SOURCE_ARTIFACT_DIR:-outputs/transformer}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-e310_smooth_morph_mild_snr_cls_raw}"
SQI_OUT_DIR="${SQI_OUT_DIR:-outputs/transformer_e310_smooth_morph_mild_snr_sqi_ml_three_class}"

echo "ARTIFACT_DIR=$ARTIFACT_DIR"
echo "SOURCE_ARTIFACT_DIR=$SOURCE_ARTIFACT_DIR"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"

python -u -m src.transformer_pipeline.noise.synthesize_morph_damage_triplet \
  --force \
  --verbose \
  --artifact_dir "$ARTIFACT_DIR" \
  --source_artifact_dir "$SOURCE_ARTIFACT_DIR" \
  --label_version e310_smooth_morph_mild_snr \
  --noise_kinds em,ma,mix \
  --group_retries "${GROUP_RETRIES:-16}" \
  --max_train_clean "${MAX_TRAIN_CLEAN:-4000}" \
  --max_val_clean "${MAX_VAL_CLEAN:-800}" \
  --max_test_clean "${MAX_TEST_CLEAN:-800}"

python -u -m src.transformer_pipeline.noise.make_rr_noise_level \
  --force \
  --verbose \
  --artifact_dir "$ARTIFACT_DIR"

python -u -m src.transformer_pipeline.diagnostics.viz_morph_triplet_samples \
  --artifact_dir "$ARTIFACT_DIR" \
  --prefix e310 \
  --split test \
  --triplets 8 \
  --examples_per_cell 2

python -u -m src.transformer_pipeline.sqi_ml_multiclass \
  --force \
  --verbose \
  --transformer_artifact_dir "$ARTIFACT_DIR" \
  --out_dir "$SQI_OUT_DIR"

python -u -m src.transformer_pipeline.run_transformer_all \
  --force \
  --verbose \
  --stage model \
  --artifact_dir "$ARTIFACT_DIR" \
  --seed "${SEED:-0}" \
  --experiment_name "$EXPERIMENT_NAME" \
  --epochs "${EPOCHS:-24}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --lr "${LR:-6e-5}" \
  --lr_eta_min "${LR_ETA_MIN:-4e-6}" \
  --weight_decay "${WEIGHT_DECAY:-0.03}" \
  --dropout "${DROPOUT:-0.1}" \
  --cls_pool cls \
  --input_mode raw \
  --lambda_rank 0 \
  --select_best_by val_acc

python - "$ARTIFACT_DIR" "$SQI_OUT_DIR" "$EXPERIMENT_NAME" <<'PY'
import json
import sys
from pathlib import Path

artifact_dir = Path(sys.argv[1])
sqi_out_dir = Path(sys.argv[2])
experiment_name = sys.argv[3]
summary = json.loads((artifact_dir / "datasets" / "morph_damage_triplet_summary.json").read_text())
sqi = json.loads((sqi_out_dir / "three_class_summary.json").read_text())
test_report = json.loads((artifact_dir / "models" / experiment_name / "test_report.json").read_text())

def metric_table(metric: str) -> list[str]:
    rows = ["| Class | N | Mean | P10 | P50 | P90 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    data = summary["audit"]["class_metric_summary"][metric]
    for cls in ["good", "medium", "bad"]:
        d = data[cls]
        rows.append(f"| {cls} | {int(d['n'])} | {d['mean']:.4f} | {d['p10']:.4f} | {d['p50']:.4f} | {d['p90']:.4f} |")
    return rows

lines = [
    "# E3.10 Smooth-Morph Mild-SNR Audit",
    "",
    f"Artifact: `{artifact_dir}`",
    "",
    "Design:",
    "",
    "- noise kinds: `em, ma, mix`",
    "- no `bw` in the synthetic benchmark",
    "- target SNR ranges: good `10-12 dB`, medium `7-10 dB`, bad `5-8 dB`",
    "- label axis: E3.9a `smooth_morph_score`",
    "- bad guard: severe QRS/correlation triggers require `smooth_morph_score >= 0.32`",
    "- model: CLS raw transformer, no local/noise/SQI/teacher/rank heads",
    "",
    "## Counts",
    "",
    "```json",
    json.dumps(summary["split_y_class_counts"], indent=2),
    "```",
    "",
    "Noise-kind counts:",
    "",
    "```json",
    json.dumps(summary["split_noise_kind_counts"], indent=2),
    "```",
    "",
    "## measured_snr_db By Class",
    "",
    *metric_table("measured_snr_db"),
    "",
    "## smooth_morph_score By Class",
    "",
    *metric_table("smooth_morph_score"),
    "",
    "## qrs_nprd By Class",
    "",
    *metric_table("qrs_nprd"),
    "",
    "## tst_nprd By Class",
    "",
    *metric_table("tst_nprd"),
    "",
    "## beat_corr By Class",
    "",
    *metric_table("beat_corr"),
    "",
    "## SQI Baselines",
    "",
    "| Model | Test Acc | Balanced Acc | Macro F1 | Medium Recall |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for key, label in [("svm_rbf", "SQI-SVM"), ("mlp", "SQI-MLP")]:
    m = sqi["metrics"][key]["test"]
    lines.append(
        f"| {label} | {m['acc']:.4f} | {m['balanced_acc']:.4f} | {m['macro_f1']:.4f} | {m['per_class_recall']['medium']:.4f} |"
    )
lines.extend([
    "",
    "## Transformer",
    "",
    f"- test acc: `{test_report['acc']:.4f}`",
    f"- confusion matrix: `{test_report['confusion_matrix_3x3']}`",
    "",
    "Per noise-kind recall:",
    "",
    "```json",
    json.dumps(test_report.get("per_noise_kind", {}), indent=2),
    "```",
    "",
    "## Figures",
    "",
    "- `figs_label_samples/e310_counterfactual_triplets.png`",
    "- `figs_label_samples/e310_class_noise_examples.png`",
])
(artifact_dir / "e310_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {artifact_dir / 'e310_audit_report.md'}")
PY
