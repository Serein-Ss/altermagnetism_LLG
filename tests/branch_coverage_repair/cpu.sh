#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=tests/branch_coverage_repair/logs/ref_%A_%a.out
#SBATCH --error=tests/branch_coverage_repair/logs/ref_%A_%a.err
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export NUMBA_NUM_THREADS=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
bash ../run_zrs_mag.sh -m tests.branch_coverage_repair.run "$1" --index "${SLURM_ARRAY_TASK_ID:-0}"
