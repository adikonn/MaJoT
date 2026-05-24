#!/bin/bash
#SBATCH --job-name=majot_bench
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG="${CONFIG:-configs/dual_stream_rowcol_hpc.yaml}"
CKPT="${CKPT:-checkpoints/dual_stream_rowcol_hpc_v1/best_model.pt}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$PROJECT_ROOT/.venv/bin/activate}"

cd "$PROJECT_ROOT"
mkdir -p logs results
source "$VENV_ACTIVATE"

python scripts/compare_predictors.py \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --output "results/compare_${SLURM_JOB_ID}.csv"
