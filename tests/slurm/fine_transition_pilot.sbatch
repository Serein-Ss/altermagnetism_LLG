#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=tests/logs/fine_transition_pilot/%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR="${SLURM_SUBMIT_DIR}/tests/.numba_cache"
bash ../run_zrs_mag.sh -m tests.experiments.fine_transition_pilot
