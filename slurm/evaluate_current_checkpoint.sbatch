#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=altermagnetism_LLG/slurm/logs/%x-%j-evaluate-current.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME

CALIBRATION=(
  altermagnetism_LLG/data/standard_v2/train/d_wave_altermagnet_L16.h5
  altermagnetism_LLG/data/standard_v2/train/d_wave_altermagnet_L32.h5
  altermagnetism_LLG/data/standard_v2/train/d_wave_altermagnet_L64.h5
)
INPUTS=(
  altermagnetism_LLG/data/standard_v2/train/d_wave_altermagnet_L16.h5
  altermagnetism_LLG/data/standard_v2/train/d_wave_altermagnet_L32.h5
  altermagnetism_LLG/data/standard_v2/train/d_wave_altermagnet_L64.h5
  altermagnetism_LLG/data/standard_v2/test_large/d_wave_altermagnet_L96.h5
)
SIZES=(16 32 64 96)
FLOW=altermagnetism_LLG/outputs/production_v1/flow/best_checkpoint.pt
BASELINE=altermagnetism_LLG/outputs/production_v1/baseline/best_checkpoint.pt
OUT=altermagnetism_LLG/outputs/diagnostics_v2/current_checkpoint_128

bash run_zrs_mag.sh -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
bash run_zrs_mag.sh -m pytest   altermagnetism_LLG/scripts/tests/test_path_evaluation.py   altermagnetism_LLG/scripts/tests/test_size_equivariant_model.py   altermagnetism_LLG/scripts/tests/test_scalable_path_dataset.py -q

for index in "${!SIZES[@]}"; do
  bash run_zrs_mag.sh -m altermagnetism_LLG.model.evaluate     --input "${INPUTS[$index]}"     --calibration-input "${CALIBRATION[@]}"     --flow-checkpoint "$FLOW"     --baseline-checkpoint "$BASELINE"     --integration-steps 128     --device cuda     --output-dir "$OUT/L${SIZES[$index]}"
done
