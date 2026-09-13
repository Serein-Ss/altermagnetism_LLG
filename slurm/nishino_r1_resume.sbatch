#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG/logs/literature_reproduction/r1_%j.log
set -euo pipefail
cd "${1:?project root required}"
export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
bash ../run_zrs_mag.sh scripts/workflow/resume_nishino_primary.py --decision "${2:?acceptance record required}"
