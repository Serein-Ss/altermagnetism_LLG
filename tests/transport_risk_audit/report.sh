#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=tests/transport_risk_audit/logs/report.sh_%A_%a.out
#SBATCH --error=tests/transport_risk_audit/logs/report.sh_%A_%a.err
set -euo pipefail
cd /share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
bash ../run_zrs_mag.sh -m tests.transport_risk_audit.run "$@" --index "${SLURM_ARRAY_TASK_ID:-0}"
