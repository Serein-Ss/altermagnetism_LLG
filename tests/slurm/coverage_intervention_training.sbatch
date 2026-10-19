#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --array=0-2
#SBATCH --output=tests/logs/coverage_intervention/train_%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
seeds=(11 22 33)
bash ../run_zrs_mag.sh -m tests.experiments.coverage_intervention_training --seed "${seeds[$SLURM_ARRAY_TASK_ID]}" --device cuda
