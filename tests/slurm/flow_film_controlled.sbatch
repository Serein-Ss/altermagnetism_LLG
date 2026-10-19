#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --output=tests/logs/flow_film_controlled/%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
seeds=(11 11 22 22 33 33)
arms=(control flow_film flow_film control control flow_film)
bash ../run_zrs_mag.sh -m tests.experiments.flow_film_controlled --seed "${seeds[$SLURM_ARRAY_TASK_ID]}" --arm "${arms[$SLURM_ARRAY_TASK_ID]}" --device cuda
