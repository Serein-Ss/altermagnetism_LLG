#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=altermagnetism_LLG/logs/slurm/%x-%j-diagnose-current.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME

CALIBRATION=(
  altermagnetism_LLG/data/datasets/standard_v2/train/d_wave_altermagnet_L16.h5
  altermagnetism_LLG/data/datasets/standard_v2/train/d_wave_altermagnet_L32.h5
  altermagnetism_LLG/data/datasets/standard_v2/train/d_wave_altermagnet_L64.h5
)

bash run_zrs_mag.sh -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
bash run_zrs_mag.sh -m altermagnetism_LLG.scripts.analysis.diagnose_flow   --input altermagnetism_LLG/data/datasets/standard_v2/train/d_wave_altermagnet_L16.h5   --calibration-input "${CALIBRATION[@]}"   --checkpoint altermagnetism_LLG/output/production/production_v1/flow/best_checkpoint.pt   --integration-steps 16 32 64 128   --integration-seeds 4   --diversity-seeds 16   --condition-id 1   --device cuda   --output-dir altermagnetism_LLG/output/diagnostics/diagnostics_v2/ablations
