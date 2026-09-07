#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=altermagnetism_LLG/slurm/logs/%x-%j-assess-L96-3ps.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME

CALIBRATION=(
  altermagnetism_LLG/data/standard_v2_extended_2ps/train/d_wave_altermagnet_L16.h5
  altermagnetism_LLG/data/standard_v2_extended_2ps/train/d_wave_altermagnet_L32.h5
  altermagnetism_LLG/data/standard_v2_extended_2ps/train/d_wave_altermagnet_L64.h5
)
EVALUATION=(
  altermagnetism_LLG/data/standard_v2_extended_3ps/test_large/d_wave_altermagnet_L96.h5
)

bash run_zrs_mag.sh \
  altermagnetism_LLG/scripts/validation/assess_extended_observation.py \
  --calibration-files "${CALIBRATION[@]}" \
  --evaluation-files "${EVALUATION[@]}" \
  --output altermagnetism_LLG/outputs/diagnostics_v2/l96_3ps_preview_assessment.json
