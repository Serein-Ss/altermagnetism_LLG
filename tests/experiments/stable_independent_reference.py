"""Frozen CPU reference batch, with complete-path and provenance preservation."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import save,digest
from tests.experiments.independent_reference import generate,validate_inputs

CONFIG=Path('tests/conf/stable_independent_reference.json')
ROOT=Path('tests/results/stable_independent_reference')
RAW=Path('tests/data/stable_independent_reference')
FILES=[__file__,'tests/experiments/independent_reference.py','scripts/core/bauer_fast.py','scripts/workflow/bauer_campaign.py']


def cases(cfg):
    rows=[]
    for theta in cfg['theta']:
        for family in range(4):rows.append(dict(theta=theta,family=family,method='mt',dt=.02,role='primary'))
        for method in ('mt','heun'):rows.append(dict(theta=theta,family=0,method=method,dt=.01,role='numerical_control'))
    for index,row in enumerate(rows):
        row.update(case_id=index,noise_start=cfg['noise_base']+index*10000)
    order=np.random.default_rng(cfg['schedule_seed']).permutation(len(rows))
    return [rows[i] for i in order]


def prepare():
    if ROOT.exists() or RAW.exists():raise FileExistsError('preserve previous batch')
    cfg=json.loads(CONFIG.read_text());initials=np.load(cfg['initials'])
    if digest(cfg['initials'])!=cfg['initial_sha256'] or initials.shape!=(4,25,3):raise ValueError('initial bank changed')
    frozen=json.loads(Path(cfg['model_freeze']).read_text())
    for seed,expected in frozen['checkpoint_sha256'].items():
        if digest(f'tests/output/stable_window_learning/window11/{seed}/step_40000.pt')!=expected:raise ValueError('candidate changed')
    rows=cases(cfg);all_noise=[]
    for row in rows:
        noise=row['noise_start']+np.arange(2*cfg['n_per_half']);all_noise.extend(noise.tolist())
        validate_inputs(np.repeat(initials[row['family']:row['family']+1],len(noise),axis=0),noise,
            **{k:cfg[k] for k in ('duration','save_dt','anisotropy','damping')},theta=row['theta'],dt=row['dt'],method=row['method'])
    if len(set(all_noise))!=len(all_noise) or len(all_noise)!=cfg['budget']['total_paths']:raise ValueError('noise or budget mismatch')
    timing=json.loads(Path('tests/results/reference_runtime_pilot/summary.json').read_text())
    if timing['status']!='COMPLETE_RUNTIME_PILOT_NOT_CERTIFIED':raise ValueError('timing incomplete')
    ROOT.mkdir(parents=True);RAW.mkdir(parents=True)
    save(ROOT/'manifest.json',dict(status='FROZEN_FOR_GENERATION',config=cfg,config_sha256=digest(CONFIG),
        model_freeze_sha256=digest(cfg['model_freeze']),code_sha256={p:digest(p) for p in FILES},cases=rows,
        total_paths=len(all_noise),seed_provenance='New840000000 namespace, disjoint from pilot200/201million, first-loop400/401million, runtime830million. Counter namespaces do not by themselves prove RNG independence.',
        timing_source_sha256=digest('tests/results/reference_runtime_pilot/summary.json')))


def run(index):
    if not os.getenv('SLURM_JOB_ID'):raise RuntimeError('CPU Slurm allocation required')
    manifest=json.loads((ROOT/'manifest.json').read_text());cfg=manifest['config']
    if not 0<=index<len(manifest['cases']):raise ValueError('unregistered array index')
    if digest(CONFIG)!=manifest['config_sha256'] or digest(cfg['model_freeze'])!=manifest['model_freeze_sha256']:
        raise ValueError('frozen configuration changed')
    for path,expected in manifest['code_sha256'].items():
        if digest(path)!=expected:raise ValueError(f'frozen source changed: {path}')
    if digest(cfg['initials'])!=cfg['initial_sha256']:raise ValueError('initial bank changed')
    case=manifest['cases'][index];case_id=case['case_id']
    out=ROOT/f'case_{case_id}';path=RAW/f'case_{case_id}.npz'
    if out.exists() or path.exists():raise FileExistsError('no silent retries or overwrites')
    out.mkdir();initials=np.load(cfg['initials']);n=cfg['n_per_half']
    states=np.repeat(initials[case['family']:case['family']+1],2*n,axis=0)
    noise=case['noise_start']+np.arange(2*n)
    report=dict(status='RUNNING',case=case,job_id=os.getenv('SLURM_JOB_ID'),manifest_sha256=digest(ROOT/'manifest.json'))
    save(out/'summary.json',report);begin=time.monotonic()
    paths,numerics=generate(states,noise,**{k:cfg[k] for k in ('duration','save_dt','anisotropy','damping')},
        theta=case['theta'],dt=case['dt'],method=case['method'])
    np.savez_compressed(path,spins=paths,initial=states[0],seeds=noise,half=np.repeat([0,1],n),
        reduced_time=np.arange(paths.shape[1])*cfg['save_dt'])
    report.update(status='COMPLETE_NUMERICAL_REFERENCE_NOT_CERTIFIED',seconds=time.monotonic()-begin,
        numerics=numerics,data_sha256=digest(path),r1_paths=n,r2_paths=n)
    save(out/'summary.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--index',type=int)
    args=p.parse_args();prepare() if args.prepare else run(args.index)
