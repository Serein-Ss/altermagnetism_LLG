#!/usr/bin/env bash
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum -c tests/miltat_refinement_20260920/submission.sha256
bash ../run_zrs_mag.sh -m tests.miltat_refinement_20260920.run --device cuda --index "$SLURM_ARRAY_TASK_ID"
