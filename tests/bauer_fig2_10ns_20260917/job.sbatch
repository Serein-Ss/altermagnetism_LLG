#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG/tests/bauer_fig2_10ns_20260917/logs/%j.out
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG/tests/bauer_fig2_10ns_20260917/logs/%j.err
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export NUMBA_NUM_THREADS=3
export NUMBA_CACHE_DIR="$PWD/tests/bauer_fig2_10ns_20260917/numba_cache"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
sha256sum -c tests/bauer_fig2_10ns_20260917/submission.sha256
bash ../run_zrs_mag.sh -m tests.bauer_fig2_10ns_20260917.run
