#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
cd "${1:?project root}"
export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 OPENBLAS_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
bash ../run_zrs_mag.sh scripts/config/validate_literature_configs.py
bash ../run_zrs_mag.sh -m pytest scripts/tests -q
