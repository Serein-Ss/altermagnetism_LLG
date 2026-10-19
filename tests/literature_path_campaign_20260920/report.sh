#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=tests/literature_path_campaign_20260920/logs/report_%j.out
#SBATCH --error=tests/literature_path_campaign_20260920/logs/report_%j.err
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
bash ../run_zrs_mag.sh -m tests.literature_path_campaign_20260920.report
bash ../run_zrs_mag.sh -m tests.progress_dashboard.build
bash ../run_zrs_mag.sh -m unittest tests.progress_dashboard.test_page
bash ../run_zrs_mag.sh -m tests.progress_dashboard.offline
