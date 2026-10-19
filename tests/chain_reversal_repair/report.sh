#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=tests/chain_reversal_repair/logs/report_%A_%a.out
#SBATCH --error=tests/chain_reversal_repair/logs/report_%A_%a.err
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
bash ../run_zrs_mag.sh -m tests.chain_reversal_repair.report "$@" --index "${SLURM_ARRAY_TASK_ID:-0}"
