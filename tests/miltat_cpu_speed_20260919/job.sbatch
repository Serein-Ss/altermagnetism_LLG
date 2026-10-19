#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=tests/miltat_cpu_speed_20260919/logs/%j.out
#SBATCH --error=tests/miltat_cpu_speed_20260919/logs/%j.err
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
sha256sum -c tests/miltat_cpu_speed_20260919/submission.sha256
bash ../run_zrs_mag.sh -m tests.miltat_cpu_speed_20260919.run
