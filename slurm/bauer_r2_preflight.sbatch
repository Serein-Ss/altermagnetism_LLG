#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/share/home/xlzou/WORKSPACE/rszhong/workspace/test_new_project/altermagnetism_LLG/logs/literature_reproduction/r2_preflight_%j.log
set -euo pipefail
cd "${1:?project root}"
SNAPSHOT="${2:?frozen source root}"
export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 LLG_CODE_COMMIT=8df33aca80fa0b6dfe943804400398dad5d31470
bash ../run_zrs_mag.sh "${SNAPSHOT}/scripts/literature/bauer_2011/preflight.py" \
    --config "${SNAPSHOT}/conf/literature/bauer_2011.yaml" \
    --project-root "$PWD" --run-id 20260910_8df33ac_060e74d3
