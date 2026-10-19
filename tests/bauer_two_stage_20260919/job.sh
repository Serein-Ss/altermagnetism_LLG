#!/usr/bin/env bash
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export NUMBA_CACHE_DIR="$PWD/tests/bauer_two_stage_20260919/numba_cache"
sha256sum -c tests/bauer_two_stage_20260919/submission.sha256
sizes=(16 25 32 50 100)
module=tests.bauer_two_stage_20260919
case "$1" in
  prepare) bash ../run_zrs_mag.sh -m "$module.prepare" --length "${sizes[$SLURM_ARRAY_TASK_ID]}" ;;
  calibrate) bash ../run_zrs_mag.sh -m "$module.prepare" --calibrate ;;
  smoke)
    bash ../run_zrs_mag.sh -m "$module.test_contracts"
    for index in 0 3 6; do bash ../run_zrs_mag.sh -m "$module.train" --index "$index" --device cuda --smoke; done ;;
  train) bash ../run_zrs_mag.sh -m "$module.train" --index "$SLURM_ARRAY_TASK_ID" --device cuda ;;
  generate) bash ../run_zrs_mag.sh -m "$module.evaluate" generate --device cuda ;;
  reference) bash ../run_zrs_mag.sh -m "$module.evaluate" reference --length "${sizes[$SLURM_ARRAY_TASK_ID]}" ;;
  assess) bash ../run_zrs_mag.sh -m "$module.evaluate" assess ;;
  long) bash ../run_zrs_mag.sh -m "$module.evaluate" long --device cuda ;;
  report)
    bash ../run_zrs_mag.sh -m "$module.report"
    bash ../run_zrs_mag.sh -m tests.progress_dashboard.build
    bash ../run_zrs_mag.sh -m tests.progress_dashboard.offline
    bash ../run_zrs_mag.sh -m tests.progress_dashboard.check_delivery ;;
  *) exit 2 ;;
esac
