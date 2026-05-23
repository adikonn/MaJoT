#!/bin/bash
#SBATCH --job-name=majot_dual_stream
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Шаблон SLURM для суперкомпьютера ВК.
# Перед первым запуском отредактируйте пути и при необходимости модули/партицию.
#
# Пример:
#   mkdir -p logs
#   export WANDB_API_KEY=...
#   sbatch scripts/hpc/submit_train.slurm

set -euo pipefail

# --- Настройки (изменить под кластер) ---
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
CONFIG="${CONFIG:-configs/dual_stream_rowcol_hpc.yaml}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$PROJECT_ROOT/.venv/bin/activate}"

cd "$PROJECT_ROOT"
mkdir -p logs

# module load cuda/12.1  # раскомментировать при необходимости
# module load python/3.12

source "$VENV_ACTIVATE"

echo "=== MaJoT train ==="
echo "Host: $(hostname)"
echo "Date: $(date -Is)"
echo "Config: $CONFIG"
nvidia-smi || true

# Кэш датасета (если ещё нет на общем диске)
python scripts/prepare_dataset.py --config "$CONFIG"

python -u scripts/train.py \
  --config "$CONFIG" \
  --override wandb.mode=online

echo "=== Готово. Чекпойнты: см. checkpoint_dir в конфиге ==="
