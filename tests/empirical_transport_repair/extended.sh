#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=tests/empirical_transport_repair/logs/extended_%A_%a.out
#SBATCH --error=tests/empirical_transport_repair/logs/extended_%A_%a.err
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
case "$1" in
  supervision_smoke) bash ../run_zrs_mag.sh -m tests.empirical_transport_repair.supervision smoke --index "$SLURM_ARRAY_TASK_ID" ;;
  supervision_train) bash ../run_zrs_mag.sh -m tests.empirical_transport_repair.supervision train --index "$SLURM_ARRAY_TASK_ID" ;;
  resample_smoke) bash ../run_zrs_mag.sh -m tests.empirical_transport_repair.resample --index "${SLURM_ARRAY_TASK_ID:-0}" --smoke ;;
  resample) bash ../run_zrs_mag.sh -m tests.empirical_transport_repair.resample --index "$SLURM_ARRAY_TASK_ID" ;;
  *) exit 2 ;;
esac
