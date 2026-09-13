#!/usr/bin/env bash
set -euo pipefail
cd "${1:?project root}"
manifest=${2:?review manifest}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=GNU PYTHONDONTWRITEBYTECODE=1
exec bash ../run_zrs_mag.sh -m scripts.analysis.bauer_revision4_audit --manifest "$manifest"
