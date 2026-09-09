#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=allcpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=altermagnetism_LLG/logs/slurm/%x-%j-aggregate-v2.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME

bash run_zrs_mag.sh altermagnetism_LLG/scripts/analysis/aggregate_seed_evaluations.py \
  --root altermagnetism_LLG/output/production/production_v2 \
  --seeds 20260908 20260909 20260910 \
  --sizes 16 32 64 96 \
  --output altermagnetism_LLG/output/production/production_v2/three_seed_summary.json
