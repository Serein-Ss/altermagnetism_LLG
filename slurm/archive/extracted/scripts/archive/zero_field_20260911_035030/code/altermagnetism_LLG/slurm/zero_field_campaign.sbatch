#!/usr/bin/env bash
# Allocation, arrays and dependencies are supplied by zero_field_campaign.py.
set -euo pipefail
cd "${1:?project root}"
campaign="${2:?frozen campaign}"
mode="${3:?worker mode}"
export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
# Exec preserves the batch PID so Slurm's warning reaches the worker.
snapshot="${BASH_SOURCE[0]%/slurm/zero_field_campaign.sbatch}"
exec bash ../run_zrs_mag.sh "$snapshot/scripts/workflow/zero_field_campaign.py" --campaign "$campaign" --worker "$mode"

