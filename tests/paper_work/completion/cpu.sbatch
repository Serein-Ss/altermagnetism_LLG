#!/bin/bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=tests/paper_work/completion/logs/cpu_%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMBA_CACHE_DIR="${SLURM_SUBMIT_DIR}/tests/paper_work/completion/logs/numba_${SLURM_JOB_ID}"
if [[ "$1" == analysis ]]; then
  exec bash ../run_zrs_mag.sh -m tests.paper_work.completion.analyze
fi
exec bash ../run_zrs_mag.sh -m tests.paper_work.completion.run "$1" --index "${SLURM_ARRAY_TASK_ID:-0}"
