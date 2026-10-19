#!/usr/bin/env bash
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum -c tests/bauer_closed_loop_20260919/submission.sha256
case "$1" in
  audit)
    bash ../run_zrs_mag.sh -m tests.bauer_closed_loop_20260919.test_loop
    bash ../run_zrs_mag.sh tests/mdgen_path_refactor_20260918/test_data.py
    bash ../run_zrs_mag.sh -m tests.bauer_closed_loop_20260919.audit
    ;;
  smoke)
    bash ../run_zrs_mag.sh -m tests.bauer_closed_loop_20260919.run --smoke --device cuda
    ;;
  train)
    bash ../run_zrs_mag.sh -m tests.bauer_closed_loop_20260919.run --index "$SLURM_ARRAY_TASK_ID" --device cuda
    ;;
  report)
    bash ../run_zrs_mag.sh -m tests.bauer_closed_loop_20260919.report
    ;;
  *) exit 2 ;;
esac
