#!/usr/bin/env bash
# Run project commands exclusively through the dedicated Conda interpreter.
set -euo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
exec /share/home/xlzou/Anaconda3/envs/zrs-mag/bin/python "$@"
