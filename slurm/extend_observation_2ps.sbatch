#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=altermagnetism_LLG/slurm/logs/%x-%A_%a-extend-2ps.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME

INPUTS=(
  altermagnetism_LLG/data/standard_v2/train/d_wave_altermagnet_L16.h5
  altermagnetism_LLG/data/standard_v2/train/d_wave_altermagnet_L32.h5
  altermagnetism_LLG/data/standard_v2/train/d_wave_altermagnet_L64.h5
  altermagnetism_LLG/data/standard_v2/test_large/d_wave_altermagnet_L96.h5
)
OUTPUTS=(
  altermagnetism_LLG/data/standard_v2_extended_2ps/train/d_wave_altermagnet_L16.h5
  altermagnetism_LLG/data/standard_v2_extended_2ps/train/d_wave_altermagnet_L32.h5
  altermagnetism_LLG/data/standard_v2_extended_2ps/train/d_wave_altermagnet_L64.h5
  altermagnetism_LLG/data/standard_v2_extended_2ps/test_large/d_wave_altermagnet_L96.h5
)

bash run_zrs_mag.sh altermagnetism_LLG/scripts/datasets/extend_scalable_hdf5.py   --input "${INPUTS[$SLURM_ARRAY_TASK_ID]}"   --output "${OUTPUTS[$SLURM_ARRAY_TASK_ID]}"   --target-duration-ps 2.0   --device cpu
