#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=tests/bauer_repair_20260924/logs/%A_%a.out
#SBATCH --error=tests/bauer_repair_20260924/logs/%A_%a.err
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
if [[ "${1:-}" == benchmark ]]; then
    bash ../run_zrs_mag.sh -m tests.bauer_repair_20260924.run benchmark --device cuda
else
    bash ../run_zrs_mag.sh -m tests.bauer_repair_20260924.run run --device cuda --index "$SLURM_ARRAY_TASK_ID"
fi
