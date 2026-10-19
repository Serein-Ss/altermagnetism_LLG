#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=tests/literature_preflight_20260917/logs/%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
systems=(bauer miltat hirst)
bash ../run_zrs_mag.sh -m tests.literature_preflight_20260917.run --system "${systems[$SLURM_ARRAY_TASK_ID]}" --device cuda
