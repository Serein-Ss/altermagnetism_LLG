#!/usr/bin/env bash
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
if [[ "$1" == smoke ]]; then
  bash ../run_zrs_mag.sh -m tests.bauer_moment_training_20260920.evaluate --index 0 --device cuda --smoke
  bash ../run_zrs_mag.sh -m tests.bauer_moment_training_20260920.evaluate --index 3 --device cuda --smoke
else
  sha256sum -c tests/bauer_moment_training_20260920/submission.sha256
  bash ../run_zrs_mag.sh -m tests.bauer_moment_training_20260920.evaluate --index "$SLURM_ARRAY_TASK_ID" --device cuda
fi
