#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=04:00:00
#SBATCH --output=tests/logs/committor_pilot/cpu_%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
if [[ "$1" == analysis ]]; then
    bash ../run_zrs_mag.sh -m tests.experiments.committor_pilot_analysis
else
    bash ../run_zrs_mag.sh -m tests.experiments.committor_pilot "$1" --index "${SLURM_ARRAY_TASK_ID:-0}"
fi
