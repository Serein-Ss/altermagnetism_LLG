"""Bounded CPU timing pilot; trajectories are never validation/test samples."""
import os
import time
from pathlib import Path
import numpy as np
from numba import get_num_threads
from scripts.workflow.bauer_campaign import initial_states,save,digest
from tests.experiments.independent_reference import generate


def main():
    if not os.getenv('SLURM_JOB_ID'):raise RuntimeError('CPU Slurm allocation required')
    destination=Path('tests/results/reference_runtime_pilot')
    raw=Path('tests/data/reference_runtime_pilot')
    if destination.exists() or raw.exists():raise FileExistsError('preserve earlier pilot')
    destination.mkdir(parents=True);raw.mkdir(parents=True)
    states=np.repeat(initial_states(4,25,'reference_runtime_pilot_20260914'),2,axis=0)
    report=dict(status='RUNNING',job_id=os.getenv('SLURM_JOB_ID'),threads=get_num_threads(),
        scope='Timing and numerical-output preflight only. Eight paths per arm, four initial families. No event precision, integrator equivalence or model certification. Exclude this entire seed/initial namespace from subsequent tests.',
        code_sha256={p:digest(p) for p in [__file__,'tests/experiments/independent_reference.py','scripts/core/bauer_fast.py']},rows=[])
    save(destination/'summary.json',report)
    for index,(method,dt) in enumerate((('mt',.02),('mt',.01),('heun',.02),('heun',.01))):
        cfg=dict(dt=dt,duration=800.,save_dt=80.,anisotropy=.1,damping=.1,theta=.13,method=method)
        noise=830000000+index*10000+np.arange(len(states))
        begin=time.monotonic()
        generate(states[:1],noise[:1]+1000,**dict(cfg,duration=.1,save_dt=.1))
        warmup=time.monotonic()-begin
        begin=time.monotonic();paths,manifest=generate(states,noise,**cfg)
        elapsed=time.monotonic()-begin
        path=raw/f'{method}_dt{dt}.npz'
        np.savez_compressed(path,spins=paths,initial=states,seeds=noise)
        report['rows'].append(dict(method=method,dt=dt,warmup_seconds=warmup,simulation_seconds=elapsed,
            manifest=manifest,sha256=digest(path),
            projected_seconds_256_paths_duration16000_same_threads=elapsed*(256/len(states))*(16000/800),
            projection_scope='Linear work extrapolation at unchanged threads/hardware, not a promised runtime; excludes compilation/I/O/queue time.'))
        save(destination/'summary.json',report)
    report['status']='COMPLETE_RUNTIME_PILOT_NOT_CERTIFIED';save(destination/'summary.json',report)


if __name__=='__main__':main()
