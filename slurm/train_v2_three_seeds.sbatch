#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=altermagnetism_LLG/slurm/logs/%x-%A_%a-train-v2.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME

SEEDS=(20260908 20260909 20260910)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
TRAIN_FILES=(
  altermagnetism_LLG/data/standard_v2_extended_3ps/train/d_wave_altermagnet_L16.h5
  altermagnetism_LLG/data/standard_v2_extended_3ps/train/d_wave_altermagnet_L32.h5
  altermagnetism_LLG/data/standard_v2_extended_3ps/train/d_wave_altermagnet_L64.h5
)
OUT="altermagnetism_LLG/outputs/production_v2/seed_$SEED"

bash run_zrs_mag.sh -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
bash run_zrs_mag.sh -m pytest altermagnetism_LLG/scripts/tests -q

for MODEL in deterministic flow; do
  bash run_zrs_mag.sh -m altermagnetism_LLG.model.train     --model-type "$MODEL"     --input "${TRAIN_FILES[@]}"     --output "$OUT/$MODEL"     --epochs 50     --batch-size 1     --crop-sizes 0 0 48     --gradient-accumulation 4     --hidden 32     --blocks 4     --architecture-version 2     --patience 10     --num-workers 2     --seed "$SEED"     --device cuda
done
