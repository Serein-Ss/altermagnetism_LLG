#!/usr/bin/env bash
#SBATCH --job-name=bauer-first-loop
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
set -euo pipefail
cd '/share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG/scripts/archive/bauer_first_loop_20260913_v1/code'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=GNU
export NUMBA_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR='/share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG/output/bauer_first_loop/20260913_v1/numba_cache'
sha256sum -c first_loop.sha256 >/dev/null
bash '/share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/run_zrs_mag.sh' -m scripts.workflow.bauer_first_loop --config conf/first_loop.json --device cuda
