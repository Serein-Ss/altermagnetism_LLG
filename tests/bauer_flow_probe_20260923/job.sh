#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=tests/bauer_flow_probe_20260923/logs/%A_%a.out
#SBATCH --error=tests/bauer_flow_probe_20260923/logs/%A_%a.err
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
seeds=(11 22 33)
bash ../run_zrs_mag.sh -m tests.bauer_flow_probe_20260923.run run --device cuda --seed "${seeds[$SLURM_ARRAY_TASK_ID]}"
