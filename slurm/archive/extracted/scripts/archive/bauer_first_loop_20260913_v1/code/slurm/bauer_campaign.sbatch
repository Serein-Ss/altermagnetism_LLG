#!/usr/bin/env bash
#SBATCH --job-name=bauer-verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
snapshot=${1:?snapshot}
wrapper=${2:?environment wrapper}
config=${3:?config}
stage=${4:?stage}
device=${5:-cpu}
cd "$snapshot"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=GNU
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONDONTWRITEBYTECODE=1
output=$(bash "$wrapper" -c 'import json,sys;print(json.load(open(sys.argv[1]))["output"])' "$config")
export NUMBA_CACHE_DIR="$output/numba_cache/${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR" "$output/runner_status"
record_exit() {
    code=$?
    trap - EXIT
    bash "$wrapper" - "$output" "$stage" "$code" <<'PY'
import json,os,sys
from pathlib import Path
root,stage,code=sys.argv[1:]
p=Path(root)/'runner_status'/f"{stage}_{os.getenv('SLURM_ARRAY_TASK_ID','0')}.json"
p.write_text(json.dumps(dict(stage=stage,exit_code=int(code),job_id=os.getenv('SLURM_JOB_ID'),
    array_task_id=os.getenv('SLURM_ARRAY_TASK_ID'),device=os.getenv('CUDA_VISIBLE_DEVICES')),indent=2)+'\n')
PY
    exit "$code"
}
trap record_exit EXIT
sha256sum -c code.sha256 >/dev/null
bash "$wrapper" -m scripts.workflow.bauer_campaign --config "$config" --stage "$stage" --device "$device"
if [[ "$stage" == software ]]; then
    GATE_F_TEST_DEVICE=cuda bash "$wrapper" -m pytest -q scripts/tests/test_gate_f.py --junitxml="$output/software/gate_f_cuda_pytest.xml"
fi
