#!/usr/bin/env bash
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
case "$1" in
  reference_smoke) bash ../run_zrs_mag.sh -m tests.bauer_teacher_state_20260920.sample reference --device cpu --smoke ;;
  reference) bash ../run_zrs_mag.sh -m tests.bauer_teacher_state_20260920.sample reference --device cpu ;;
  smoke) bash ../run_zrs_mag.sh -m tests.bauer_teacher_state_20260920.sample model --device cuda --smoke ;;
  model)
    seeds=(11 22 33)
    bash ../run_zrs_mag.sh -m tests.bauer_teacher_state_20260920.sample model --device cuda --seed "${seeds[$SLURM_ARRAY_TASK_ID]}" ;;
  *) exit 2 ;;
esac
