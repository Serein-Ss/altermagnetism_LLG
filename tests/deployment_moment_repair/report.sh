#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=tests/deployment_moment_repair/logs/report.sh_%A_%a.out
#SBATCH --error=tests/deployment_moment_repair/logs/report.sh_%A_%a.err
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
bash ../run_zrs_mag.sh -m tests.deployment_moment_repair.report "$@" --index "${SLURM_ARRAY_TASK_ID:-0}"
