#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --array=0-11
#SBATCH --output=tests/logs/stable_independent_reference/%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export NUMBA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
bash ../run_zrs_mag.sh -m tests.experiments.stable_independent_reference --index "$SLURM_ARRAY_TASK_ID"
