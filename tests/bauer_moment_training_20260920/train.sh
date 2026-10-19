#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=tests/bauer_moment_training_20260920/logs/train_%A_%a.out
#SBATCH --error=tests/bauer_moment_training_20260920/logs/train_%A_%a.err
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum -c tests/bauer_transition_20260920/submission.sha256
sha256sum -c tests/bauer_moment_training_20260920/submission.sha256
bash ../run_zrs_mag.sh -m tests.bauer_moment_training_20260920.train --index "$SLURM_ARRAY_TASK_ID" --device cuda
