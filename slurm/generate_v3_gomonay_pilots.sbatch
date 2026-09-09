#!/usr/bin/env bash
#SBATCH --job-name=zrs-mag
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-59%5
#SBATCH --output=altermagnetism_LLG/logs/slurm/%x-%A_%a-v3-pilot.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME

PROTOCOL=altermagnetism_LLG/data/datasets/standard_v3/altermagnet_gomonay2024/protocol.candidate.yaml
GENERATOR=altermagnetism_LLG/scripts/generation/generate_v3_gomonay_shard.py
ROOT=altermagnetism_LLG/data/datasets/standard_v3/altermagnet_gomonay2024/convergence
TASK="$SLURM_ARRAY_TASK_ID"

bash run_zrs_mag.sh altermagnetism_LLG/scripts/validation/preflight_standard_v3.py

SIZE=64
DRIVE=0.78
DT=0.05
POL=(0.7071067811865476 0.7071067811865476 0.0)

if (( TASK < 20 )); then
  DRIVES=(0.6 0.7 0.78 0.8 0.9)
  INDEX=$((TASK / 4))
  SHARD=$((TASK % 4))
  DRIVE="${DRIVES[$INDEX]}"
  CONDITION=$((100 + INDEX))
  OUT="$ROOT/drive_duration/L64_H${DRIVE}_shard${SHARD}.h5"
elif (( TASK < 28 )); then
  DTS=(0.1 0.025)
  LOCAL=$((TASK - 20))
  INDEX=$((LOCAL / 4))
  SHARD=$((LOCAL % 4))
  DT="${DTS[$INDEX]}"
  CONDITION=$((200 + INDEX))
  OUT="$ROOT/dt/L64_dt${DT}fs_H0.78_shard${SHARD}.h5"
elif (( TASK < 44 )); then
  SIZES=(32 48 96 128)
  LOCAL=$((TASK - 28))
  INDEX=$((LOCAL / 4))
  SHARD=$((LOCAL % 4))
  SIZE="${SIZES[$INDEX]}"
  CONDITION=$((300 + INDEX))
  OUT="$ROOT/size/L${SIZE}_H0.78_shard${SHARD}.h5"
elif (( TASK < 56 )); then
  LOCAL=$((TASK - 44))
  INDEX=$((LOCAL / 4))
  SHARD=$((LOCAL % 4))
  CONDITION=$((400 + INDEX))
  case "$INDEX" in
    0) LABEL=100; POL=(1.0 0.0 0.0) ;;
    1) LABEL=010; POL=(0.0 1.0 0.0) ;;
    2) LABEL=1m10; POL=(0.7071067811865476 -0.7071067811865476 0.0) ;;
  esac
  OUT="$ROOT/orientation/L64_${LABEL}_H0.78_shard${SHARD}.h5"
else
  SHARD=$((TASK - 56))
  DRIVE=0.0
  CONDITION=500
  OUT="$ROOT/equilibrium/L64_H0_shard${SHARD}.h5"
fi

bash run_zrs_mag.sh "$GENERATOR" \
  --mode pilot \
  --protocol "$PROTOCOL" \
  --output "$OUT" \
  --size "$SIZE" \
  --temperature-k 5.0 \
  --drive-t "$DRIVE" \
  --polarization "${POL[@]}" \
  --condition-id "$CONDITION" \
  --shard-index "$SHARD" \
  --paths 32 \
  --dt-fs "$DT" \
  --duration-ps 8.0 \
  --saved-dt-fs 2.5 \
  --equilibration-ps 0.2 \
  --pulse-ps 0.5 \
  --alpha 0.01 \
  --base-seed 20260908 \
  --device cpu
