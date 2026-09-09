#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=altermagnetism_LLG/logs/slurm/%x-%j-smoke.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME

TRAIN_FILES=(
  altermagnetism_LLG/data/datasets/standard_v2/train/d_wave_altermagnet_L16.h5
)
L96=altermagnetism_LLG/data/datasets/standard_v2/test_large/d_wave_altermagnet_L96.h5
OUT=altermagnetism_LLG/output/smoke/gpu_smoke

bash run_zrs_mag.sh -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
bash run_zrs_mag.sh -m pytest altermagnetism_LLG/scripts/tests -q
bash run_zrs_mag.sh -m altermagnetism_LLG.scripts.training.train   --model-type deterministic --architecture-version 1 --input "${TRAIN_FILES[@]}"   --output "$OUT/baseline" --epochs 1 --batch-size 2 --crop-size 8   --hidden 8 --blocks 1 --patience 1 --device cuda --overwrite
bash run_zrs_mag.sh -m altermagnetism_LLG.scripts.training.train   --model-type flow --architecture-version 1 --input "${TRAIN_FILES[@]}"   --output "$OUT/flow" --epochs 1 --batch-size 2 --crop-size 8   --hidden 8 --blocks 1 --patience 1 --device cuda --overwrite
bash run_zrs_mag.sh -m altermagnetism_LLG.scripts.analysis.evaluate_l96   --input "$L96"   --calibration-input "${TRAIN_FILES[@]}"   --flow-checkpoint "$OUT/flow/best_checkpoint.pt"   --baseline-checkpoint "$OUT/baseline/best_checkpoint.pt"   --integration-steps 2 --max-paths-per-condition 1 --device cuda   --output-dir "$OUT/l96_smoke_evaluation" --overwrite
