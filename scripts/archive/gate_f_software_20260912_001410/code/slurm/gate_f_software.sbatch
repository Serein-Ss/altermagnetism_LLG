#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
snapshot=${1:?frozen source directory}
wrapper=${2:?zrs-mag wrapper}
device=${3:?cpu or cuda}
result=${4:?result directory}
cd "$snapshot"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=GNU
export PYTHONDONTWRITEBYTECODE=1
bash "$wrapper" -m scripts.validation.run_gate_f_software --device "$device" --output "$result"
