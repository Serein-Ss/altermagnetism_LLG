#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=tests/logs/path_learning_audit/software_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR="$SLURM_SUBMIT_DIR/tests/logs/path_learning_audit/numba_$SLURM_JOB_ID"
mkdir -p "$NUMBA_CACHE_DIR"
bash ../run_zrs_mag.sh -m pytest tests/unit -q --junitxml="tests/logs/path_learning_audit/software_$SLURM_JOB_ID.xml"
