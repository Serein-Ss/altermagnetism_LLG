#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --signal=B:USR1@180
set -euo pipefail
project=${1:?project root}
index=${2:?task index}
cd "$project"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=GNU PYTHONDONTWRITEBYTECODE=1
exec bash ../run_zrs_mag.sh scripts/workflow/resume_after_restructure.py --index "$index"
