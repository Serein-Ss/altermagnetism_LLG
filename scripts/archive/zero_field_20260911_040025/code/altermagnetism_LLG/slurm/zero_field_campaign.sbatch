#!/usr/bin/env bash
# Explicit paths: Slurm copies this file to its spool, so BASH_SOURCE is not the source directory.
set -euo pipefail
cd "${1:?project root}"
campaign="${2:?frozen campaign}"
mode="${3:?worker mode}"
snapshot="${4:?frozen source root}"
export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
exec bash ../run_zrs_mag.sh "$snapshot/scripts/workflow/zero_field_campaign.py" --campaign "$campaign" --worker "$mode"
