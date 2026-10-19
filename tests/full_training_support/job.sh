#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=tests/full_training_support/logs/scan_%A_%a.out
#SBATCH --error=tests/full_training_support/logs/scan_%A_%a.err
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
radii=(1 2 4)
bash ../run_zrs_mag.sh -m tests.full_training_support.run scan --radius "${radii[$SLURM_ARRAY_TASK_ID]}"
