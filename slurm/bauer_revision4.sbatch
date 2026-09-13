#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --signal=B:USR1@180
set -euo pipefail
snapshot=${1:?snapshot}
wrapper=${2:?environment wrapper}
manifest=${3:?manifest}
stage=${4:?stage}
index=${5:-0}
cd "$snapshot"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=GNU PYTHONDONTWRITEBYTECODE=1
exec bash "$wrapper" -m scripts.workflow.bauer_revision4 --manifest "$manifest" --stage "$stage" --index "$index"
