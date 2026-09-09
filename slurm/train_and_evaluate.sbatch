#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=altermagnetism_LLG/logs/slurm/%x-%j-training.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME

TRAIN_FILES=(
  altermagnetism_LLG/data/datasets/standard_v2/train/d_wave_altermagnet_L16.h5
  altermagnetism_LLG/data/datasets/standard_v2/train/d_wave_altermagnet_L32.h5
  altermagnetism_LLG/data/datasets/standard_v2/train/d_wave_altermagnet_L64.h5
)
L96=altermagnetism_LLG/data/datasets/standard_v2/test_large/d_wave_altermagnet_L96.h5
OUT=altermagnetism_LLG/output/production/production_v1

bash run_zrs_mag.sh -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
bash run_zrs_mag.sh -m pytest altermagnetism_LLG/scripts/tests -q
bash run_zrs_mag.sh -m altermagnetism_LLG.scripts.training.train   --model-type deterministic --architecture-version 1 --input "${TRAIN_FILES[@]}"   --output "$OUT/baseline" --epochs 50 --batch-size 4 --crop-size 16   --hidden 32 --blocks 4 --patience 10 --num-workers 2   --device cuda --overwrite
bash run_zrs_mag.sh -m altermagnetism_LLG.scripts.training.train   --model-type flow --architecture-version 1 --input "${TRAIN_FILES[@]}"   --output "$OUT/flow" --epochs 50 --batch-size 4 --crop-size 16   --hidden 32 --blocks 4 --patience 10 --num-workers 2   --device cuda --overwrite
bash run_zrs_mag.sh -m altermagnetism_LLG.scripts.analysis.evaluate_l96   --input "$L96"   --calibration-input "${TRAIN_FILES[@]}"   --flow-checkpoint "$OUT/flow/best_checkpoint.pt"   --baseline-checkpoint "$OUT/baseline/best_checkpoint.pt"   --integration-steps 64 --device cuda   --output-dir "$OUT/l96_evaluation" --overwrite
