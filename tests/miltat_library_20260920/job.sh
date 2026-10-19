#!/usr/bin/env bash
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum -c tests/miltat_library_20260920/submission.sha256
module=tests.miltat_library_20260920
case "$1" in
  smoke)
    bash ../run_zrs_mag.sh -m "$module.generate" cold --index 2 --device cuda --smoke
    bash ../run_zrs_mag.sh -m "$module.generate" parent --index 40 --device cuda --smoke
    bash ../run_zrs_mag.sh -m "$module.generate" finalize --smoke ;;
  cold) bash ../run_zrs_mag.sh -m "$module.generate" cold --index "$SLURM_ARRAY_TASK_ID" --device cuda ;;
  cold_gate) bash ../run_zrs_mag.sh -m "$module.quality" cold_gate ;;
  parent) bash ../run_zrs_mag.sh -m "$module.generate" parent --index "$SLURM_ARRAY_TASK_ID" --device cuda ;;
  finalize) bash ../run_zrs_mag.sh -m "$module.generate" finalize ;;
  status) bash ../run_zrs_mag.sh -m "$module.quality" status ;;
  *) exit 2 ;;
esac
