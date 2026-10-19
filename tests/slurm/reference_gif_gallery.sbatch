#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --array=0-5
#SBATCH --output=tests/logs/reference_gif_gallery/render_%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
for ((group=SLURM_ARRAY_TASK_ID*12; group<(SLURM_ARRAY_TASK_ID+1)*12; group++)); do
    bash ../run_zrs_mag.sh -m tests.experiments.reference_gif_gallery --group "$group"
done
