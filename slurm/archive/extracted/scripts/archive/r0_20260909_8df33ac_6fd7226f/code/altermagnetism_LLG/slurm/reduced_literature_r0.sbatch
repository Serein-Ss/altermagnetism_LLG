#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG/logs/literature_reproduction/r0_%j.log
#SBATCH --error=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG/logs/literature_reproduction/r0_%j.log
set -euo pipefail
PROJECT_ROOT="${1:?project root required}"
CAMPAIGN="${2:?frozen campaign required}"
SNAPSHOT="${CAMPAIGN}/code/altermagnetism_LLG"
export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
cd "${PROJECT_ROOT}"
bash ../run_zrs_mag.sh "${SNAPSHOT}/scripts/workflow/run_reduced_literature_campaign.py" \
    --project-root "${PROJECT_ROOT}" --campaign "${CAMPAIGN}" \
    --workers "${SLURM_CPUS_PER_TASK}"
