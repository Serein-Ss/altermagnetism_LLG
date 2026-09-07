#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-11
#SBATCH --output=altermagnetism_LLG/slurm/logs/%x-%A_%a-evaluate-v2.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME

SEEDS=(20260908 20260909 20260910)
INPUTS=(
  altermagnetism_LLG/data/standard_v2_extended_3ps/train/d_wave_altermagnet_L16.h5
  altermagnetism_LLG/data/standard_v2_extended_3ps/train/d_wave_altermagnet_L32.h5
  altermagnetism_LLG/data/standard_v2_extended_3ps/train/d_wave_altermagnet_L64.h5
  altermagnetism_LLG/data/standard_v2_extended_3ps/test_large/d_wave_altermagnet_L96.h5
)
SIZES=(16 32 64 96)
EVALUATION_SEED=20260911
CALIBRATION=(
  "${INPUTS[0]}"
  "${INPUTS[1]}"
  "${INPUTS[2]}"
)

TASK="$SLURM_ARRAY_TASK_ID"
SEED_INDEX=$((TASK / 4))
SIZE_INDEX=$((TASK % 4))
SEED="${SEEDS[$SEED_INDEX]}"
SIZE="${SIZES[$SIZE_INDEX]}"
ROOT="altermagnetism_LLG/outputs/production_v2/seed_$SEED"
OUT="$ROOT/evaluation/L$SIZE"

bash run_zrs_mag.sh -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
bash run_zrs_mag.sh -m altermagnetism_LLG.model.evaluate \
  --input "${INPUTS[$SIZE_INDEX]}" \
  --calibration-input "${CALIBRATION[@]}" \
  --flow-checkpoint "$ROOT/flow/best_checkpoint.pt" \
  --baseline-checkpoint "$ROOT/deterministic/best_checkpoint.pt" \
  --integration-steps 128 \
  --seed "$EVALUATION_SEED" \
  --device cuda \
  --output-dir "$OUT"
