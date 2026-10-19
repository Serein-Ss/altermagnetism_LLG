#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=tests/cross_dimension_20260924/logs/%A_%a.out
#SBATCH --error=tests/cross_dimension_20260924/logs/%A_%a.err
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
if [[ "${1:-}" == benchmark ]]; then
    bash ../run_zrs_mag.sh -m tests.cross_dimension_20260924.run benchmark --device cuda
elif [[ "${1:-}" == oracle ]]; then
    bash ../run_zrs_mag.sh -m tests.cross_dimension_20260924.run oracle --device cuda
else
    bash ../run_zrs_mag.sh -m tests.cross_dimension_20260924.run run --index "$SLURM_ARRAY_TASK_ID" --device cuda
fi
