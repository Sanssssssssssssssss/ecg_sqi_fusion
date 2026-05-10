#!/bin/bash
#SBATCH -J ptbxl_mtl
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=NONE

set -euo pipefail

# --- modules (Rocky8 ampere env) ---
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.11.0-icl

# --- project ---
cd /home/cx272/final_project/ecg_sqi_fusion

# --- venv ---
source .venv/bin/activate
mkdir -p logs

# --- quick sanity ---
echo "Host: $(hostname)"
echo "Time: $(date)"
python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda?", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for transformer training; refusing to run on CPU.")
PY

# --- run transformer stage only; preprocessing is a separate local/CPU pipeline ---
python -u -m src.transformer_pipeline.run_transformer_all --force --verbose
