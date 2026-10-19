#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=tests/logs/reference_runtime_pilot/%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export NUMBA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
bash ../run_zrs_mag.sh -m tests.experiments.reference_runtime_pilot
