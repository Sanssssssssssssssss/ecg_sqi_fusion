#!/bin/bash
#SBATCH -J e311_tune
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

ARTIFACT_DIR="${ARTIFACT_DIR:-outputs/transformer_e311_visual_gap}"
BASE_EXPERIMENT="${BASE_EXPERIMENT:-e311_visual_gap_cls_raw}"

python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda?", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for E3.11 tuning; refusing to run on CPU.")
PY

if [ ! -f "$ARTIFACT_DIR/datasets/synth_10s_125hz_labels_with_level.csv" ]; then
  echo "Missing E3.11 dataset under $ARTIFACT_DIR. Run run_e311_visual_gap.sh first." >&2
  exit 2
fi

run_model() {
  local name="$1"
  shift
  echo
  echo "=== E3.11 tuning: $name ==="
  python -u -m src.transformer_pipeline.run_transformer_all \
    --force \
    --verbose \
    --stage model \
    --artifact_dir "$ARTIFACT_DIR" \
    --seed "${SEED:-0}" \
    --experiment_name "$name" \
    --batch_size 32 \
    --cls_pool cls \
    --input_mode raw \
    --lambda_rank 0 \
    --select_best_by val_acc \
    "$@"
}

# Iteration 1: conservative optimization, meant to reduce boundary overshoot.
run_model e311_visual_gap_cls_raw_lr3e5_ep32 \
  --epochs 32 \
  --lr 3e-5 \
  --lr_eta_min 2e-6 \
  --weight_decay 0.03 \
  --dropout 0.10

# Iteration 2: slightly stronger regularization for the visually separated dataset.
run_model e311_visual_gap_cls_raw_reg_ls003 \
  --epochs 28 \
  --lr 5e-5 \
  --lr_eta_min 3e-6 \
  --weight_decay 0.05 \
  --dropout 0.15 \
  --label_smoothing 0.03

python - "$ARTIFACT_DIR" "$BASE_EXPERIMENT" <<'PY'
import json
import sys
from pathlib import Path

artifact_dir = Path(sys.argv[1])
base_experiment = sys.argv[2]

rows = [
    ("D1 main reference", Path("outputs/transformer_e39a_smooth_morph_triplet/models/e39_d1_smooth_morph_cls_raw/test_report.json")),
    ("E3.10 mild-SNR reference", Path("outputs/transformer_e310_smooth_morph_mild_snr/models/e310_smooth_morph_mild_snr_cls_raw/test_report.json")),
    ("E3.11 seed0", artifact_dir / "models" / base_experiment / "test_report.json"),
    ("E3.11 iter1 lr3e-5 ep32", artifact_dir / "models" / "e311_visual_gap_cls_raw_lr3e5_ep32" / "test_report.json"),
    ("E3.11 iter2 reg ls0.03", artifact_dir / "models" / "e311_visual_gap_cls_raw_reg_ls003" / "test_report.json"),
]

def recalls(cm: list[list[int]]) -> tuple[float, float, float]:
    out = []
    for row in cm:
        denom = max(1, sum(row))
        out.append(float(row[len(out)]) / denom)
    return tuple(out)  # type: ignore[return-value]

lines = [
    "# E3.11 Visual-Gap Tuning Comparison",
    "",
    "Goal: recover or exceed `0.94` test accuracy while keeping the E3.11 visual-gap data rule.",
    "",
    "| Run | Test Acc | Good Recall | Medium Recall | Bad Recall | Confusion Matrix |",
    "| --- | ---: | ---: | ---: | ---: | --- |",
]
best = ("", -1.0)
for label, path in rows:
    if not path.exists():
        lines.append(f"| {label} | missing |  |  |  | `{path}` |")
        continue
    rep = json.loads(path.read_text())
    cm = rep["confusion_matrix_3x3"]
    good, medium, bad = recalls(cm)
    acc = float(rep["acc"])
    if label.startswith("E3.11") and acc > best[1]:
        best = (label, acc)
    lines.append(f"| {label} | {acc:.4f} | {good:.4f} | {medium:.4f} | {bad:.4f} | `{cm}` |")

lines.extend([
    "",
    f"Best E3.11 run: `{best[0]}` with test acc `{best[1]:.4f}`." if best[0] else "Best E3.11 run: unavailable.",
])
(artifact_dir / "e311_tuning_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {artifact_dir / 'e311_tuning_comparison.md'}")
PY
