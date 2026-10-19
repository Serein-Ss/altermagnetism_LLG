#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG/tests/bauer_active_data_20260918/logs/%A_%a.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG/tests/bauer_active_data_20260918/logs/%A_%a.err
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_CACHE_DIR="$PWD/tests/bauer_active_data_20260918/numba_cache"
export PYTHONDONTWRITEBYTECODE=1
sha256sum -c tests/bauer_active_data_20260918/submission.sha256
sizes=(16 25 32)
bash ../run_zrs_mag.sh -m tests.bauer_active_data_20260918.run --length "${sizes[$SLURM_ARRAY_TASK_ID]}"
