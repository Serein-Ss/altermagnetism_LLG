#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=rtx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=tests/logs/bauer_diagnostics_20260913_v2/%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
seeds=(11 22 33)
GATE_F_TEST_DEVICE=cuda bash ../run_zrs_mag.sh -m pytest -q tests/unit/test_gate_f.py
bash ../run_zrs_mag.sh -m tests.experiments.bauer_checkpoint_diagnostics --config tests/conf/bauer_diagnostics_20260913_v2.json --device cuda --seed "${seeds[$SLURM_ARRAY_TASK_ID]}"
