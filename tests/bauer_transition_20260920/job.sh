#!/usr/bin/env bash
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum -c tests/bauer_transition_20260920/submission.sha256
module=tests.bauer_transition_20260920
case "$1" in
  smoke)
    bash ../run_zrs_mag.sh -m pytest -q tests/bauer_transition_20260920/test_contracts.py
    bash ../run_zrs_mag.sh -m "$module.run" train --index 0 --smoke --device cuda
    bash ../run_zrs_mag.sh -m "$module.run" train --index 3 --smoke --device cuda ;;
  train) bash ../run_zrs_mag.sh -m "$module.run" train --index "$SLURM_ARRAY_TASK_ID" --device cuda ;;
  evaluate) bash ../run_zrs_mag.sh -m "$module.run" evaluate --index "$SLURM_ARRAY_TASK_ID" --device cuda ;;
  report) bash ../run_zrs_mag.sh -m "$module.run" report ;;
  *) exit 2 ;;
esac
