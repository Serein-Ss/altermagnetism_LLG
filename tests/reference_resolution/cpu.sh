#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=tests/reference_resolution/logs/cpu_%A_%a.out
#SBATCH --error=tests/reference_resolution/logs/cpu_%A_%a.err
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
bash ../run_zrs_mag.sh -m tests.reference_resolution.run "$1" --index "${SLURM_ARRAY_TASK_ID:-0}"
