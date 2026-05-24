#!/bin/bash
#SBATCH -J e311f_r1seq
#SBATCH -A mphil-dis-sl2-gpu
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=NONE

set -euo pipefail

cd /home/cx272/final_project/ecg_sqi_fusion
mkdir -p logs reports

for task_id in 0 1 2 3; do
  echo "=== round1 task ${task_id} ==="
  SLURM_ARRAY_TASK_ID="${task_id}" bash slurm/tune_e311f_visual_best_round1.sh
done
