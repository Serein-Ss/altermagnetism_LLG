"""Bounded synthetic verification; writes results, never a physics certificate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.device == 'cuda' and (not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available()):
        raise RuntimeError('GPU allocation and working CUDA are mandatory; no CPU fallback')
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, GATE_F_TEST_DEVICE=args.device, OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, '-m', 'pytest', 'scripts/tests/test_gate_f.py', '-q',
               '-p', 'no:cacheprovider', '--junitxml='+str(args.output/'junit.xml')]
    started = time.monotonic()
    with (args.output/'pytest.log').open('x') as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    report = dict(scope='synthetic_gate_f_software_only', device=args.device,
                  status='pass' if result.returncode == 0 else 'fail', returncode=result.returncode,
                  elapsed_seconds=time.monotonic()-started, torch_version=str(torch.__version__),
                  production_enabled=False, training_started=False,
                  limitations=['No physical P1/P2 certificate', 'No real Gate-F training or distribution GO',
                               'CPU skips full-shape CUDA checks', 'Model architecture size in GPU smoke is 8 channels / 1 block'])
    with (args.output/'result.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
