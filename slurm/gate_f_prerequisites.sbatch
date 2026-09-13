#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --signal=B:USR1@180
set -euo pipefail
snapshot=${1:?snapshot}
wrapper=${2:?environment wrapper}
manifest=${3:?manifest}
stage=${4:?stage}
cd "$snapshot"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=GNU PYTHONDONTWRITEBYTECODE=1
bash "$wrapper" -m scripts.workflow.gate_f_prerequisites --manifest "$manifest" --stage "$stage" --index "${SLURM_ARRAY_TASK_ID:-0}"
