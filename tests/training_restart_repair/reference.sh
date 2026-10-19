#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=tests/training_restart_repair/logs/reference_%A_%a.out
#SBATCH --error=tests/training_restart_repair/logs/reference_%A_%a.err
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
bash ../run_zrs_mag.sh -m tests.training_restart_repair.data "$1" --index "${SLURM_ARRAY_TASK_ID:-0}"
