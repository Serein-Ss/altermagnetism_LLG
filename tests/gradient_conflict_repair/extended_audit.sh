#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=tests/gradient_conflict_repair/logs/extended_%A_%a.out
#SBATCH --error=tests/gradient_conflict_repair/logs/extended_%A_%a.err
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
bash ../run_zrs_mag.sh -m tests.gradient_conflict_repair.extended_audit --index "${SLURM_ARRAY_TASK_ID:-0}" --batches "${1:-8192}"
