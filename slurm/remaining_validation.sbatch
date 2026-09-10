#!/usr/bin/env bash
# Resources and dependencies are supplied by submit_remaining_validation.py.
set -euo pipefail
cd "${1:?project root}"
campaign="${2:?frozen campaign}"
kind="${3:?worker kind}"
export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
bash ../run_zrs_mag.sh "${campaign}/code/altermagnetism_LLG/scripts/workflow/submit_remaining_validation.py" \
  --campaign "$campaign" --worker "$kind"
