#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=altermagnetism_LLG/logs/slurm/%x-%j-smoke-v2-l64.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME

INPUT=altermagnetism_LLG/data/datasets/standard_v2_extended_2ps/train/d_wave_altermagnet_L64.h5
OUT=altermagnetism_LLG/output/smoke/gpu_smoke_v2_L64

bash run_zrs_mag.sh -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
for MODEL in deterministic flow; do
  bash run_zrs_mag.sh -m altermagnetism_LLG.scripts.training.train \
    --model-type "$MODEL" \
    --input "$INPUT" \
    --output "$OUT/$MODEL" \
    --epochs 1 \
    --batch-size 1 \
    --crop-sizes 48 \
    --gradient-accumulation 4 \
    --hidden 32 \
    --blocks 4 \
    --architecture-version 2 \
    --patience 1 \
    --num-workers 2 \
    --seed 20260908 \
    --device cuda \
    --overwrite
done
