#!/usr/bin/env bash
#SBATCH --job-name=zrs_data
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
set -euo pipefail
AUDIT_ROOT="$1"
STAGE="$2"
SNAPSHOT="$3"
LOG_DIR="$4"
export LLG_ARTIFACT_ROOT="$5"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME CUDA_VISIBLE_DEVICES
cd "$SNAPSHOT/altermagnetism_LLG"
RUNNER="$SNAPSHOT/run_zrs_mag.sh"
case "$STAGE" in
  preflight)
    bash "$RUNNER" -m pip freeze > "$LOG_DIR/python_environment.txt"
    lscpu > "$LOG_DIR/cpu_environment.txt"
    free -h > "$LOG_DIR/memory_environment.txt"
    bash "$RUNNER" -m pytest scripts/tests -q -p no:cacheprovider
    bash "$RUNNER" scripts/validation/validate_thermal_noise_statistics.py \
      --output "$AUDIT_ROOT/thermal_noise_statistics.json"
    ;;
  first)
    bash "$RUNNER" scripts/literature/validate_nishino_full.py \
      --task 0 --output-dir "$AUDIT_ROOT/nishino"
    ;;
  ensemble)
    PIDS=()
    FIRST=$((1 + (SLURM_ARRAY_TASK_ID - 1) * 20))
    LAST=$((FIRST + 19))
    for (( TASK=FIRST; TASK<=LAST && TASK<120; TASK++ )); do
      bash "$RUNNER" scripts/literature/validate_nishino_full.py \
        --task "$TASK" --output-dir "$AUDIT_ROOT/nishino" \
        > "$LOG_DIR/task-$TASK.out" 2>&1 &
      PIDS+=("$!")
    done
    FAILED=0
    for PID in "${PIDS[@]}"; do
      if ! wait "$PID"; then FAILED=1; fi
    done
    exit "$FAILED"
    ;;
  aggregate)
    bash "$RUNNER" scripts/analysis/audit_nishino_ensemble.py \
      --input-dir "$AUDIT_ROOT/nishino" --output "$AUDIT_ROOT/nishino_decision.json"
    ;;
  *) exit 2 ;;
esac
