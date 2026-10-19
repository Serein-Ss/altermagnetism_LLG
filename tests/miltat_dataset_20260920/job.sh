#!/usr/bin/env bash
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum -c tests/miltat_dataset_20260920/submission.sha256
case "$1" in
  smoke) bash ../run_zrs_mag.sh -m tests.miltat_dataset_20260920.convergence --device cuda --index 1 --smoke ;;
  convergence) bash ../run_zrs_mag.sh -m tests.miltat_dataset_20260920.convergence --device cuda --index "$SLURM_ARRAY_TASK_ID" ;;
  report) bash ../run_zrs_mag.sh -m tests.miltat_dataset_20260920.submit --report ;;
  *) exit 2 ;;
esac
