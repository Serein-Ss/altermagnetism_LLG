#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=altermagnetism_LLG/slurm/logs/%x-%j-extend-L96-3ps.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME

INPUT=altermagnetism_LLG/data/standard_v2_extended_2ps/test_large/d_wave_altermagnet_L96.h5
OUTPUT=altermagnetism_LLG/data/standard_v2_extended_3ps/test_large/d_wave_altermagnet_L96.h5

bash run_zrs_mag.sh altermagnetism_LLG/scripts/datasets/extend_scalable_hdf5.py   --input "$INPUT"   --output "$OUTPUT"   --target-duration-ps 3.0   --device cpu
