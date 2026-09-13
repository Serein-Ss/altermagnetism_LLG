#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${1:?project root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=GNU PYTHONDONTWRITEBYTECODE=1
bash ../run_zrs_mag.sh scripts/analysis/analyze_remaining_validation.py \
  --campaign scripts/archive/r2_r5_validation_20260910_110834 --analysis-id "${2:?analysis id}"
