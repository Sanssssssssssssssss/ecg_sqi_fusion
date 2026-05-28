#!/bin/bash
# Isolated E3.11 SQI research: multi-task loss conflict screen.
#SBATCH -J e311r2_loss
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --array=0-4%1
#SBATCH --output=outputs/experiment/e311_sqi_research/logs/%x_%A_%a.out
#SBATCH --error=outputs/experiment/e311_sqi_research/logs/%x_%A_%a.err
#SBATCH --mail-type=NONE

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.11.0-icl

cd /home/cx272/final_project/ecg_sqi_fusion
source .venv/bin/activate
mkdir -p outputs/experiment/e311_sqi_research/logs

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda?", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for E3.11 research training.")
PY

python -u -m src.experiment.e311_sqi_research.train \
  --group loss_conflict \
  --task_id "${SLURM_ARRAY_TASK_ID:-0}"

python -u -m src.experiment.e311_sqi_research.summarize
