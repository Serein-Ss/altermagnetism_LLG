#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/bauer_path_%j.out
#SBATCH --error=logs/bauer_path_%j.err
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
bash ../run_zrs_mag.sh -m scripts.training.train_bauer_path --device cuda "$@"
