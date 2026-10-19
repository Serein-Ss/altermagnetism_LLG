"""Train-state-only short numerical transition pilot; no model training."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.independent_reference import generate

CONFIG=Path('tests/conf/fine_transition_pilot.json')
ROOT=Path('tests/results/fine_transition_pilot')
RAW=Path('tests/data/fine_transition_pilot')


def select_anchors(train,conditions,cfg):
    states=[]; rows=[]
    for theta in cfg['theta']:
        selected=[]
        for parent in np.flatnonzero(np.isclose(conditions[:,0],theta)):
            if any(np.array_equal(train[parent,0],train[p,0]) for p in selected): continue
            selected.append(int(parent))
            if len(selected)==cfg['families_per_theta']: break
        if len(selected)!=cfg['families_per_theta']: raise ValueError('insufficient train initial families')
        for family,parent in enumerate(selected):
            for frame in cfg['start_frames']:
                states.append(train[parent,frame])
                rows.append(dict(parent=parent,family=family,start_frame=frame,theta=theta))
    original=np.asarray(states,dtype=np.float64)
    norm=np.linalg.norm(original,axis=-1,keepdims=True)
    if not np.isfinite(original).all() or np.max(np.abs(norm-1))>1e-5: raise ValueError('invalid train spins')
    return original,original/norm,rows


def prepare():
    cfg=json.loads(CONFIG.read_text())
    if ROOT.exists() or RAW.exists(): raise FileExistsError('preserve prior pilot')
    if digest(cfg['data'])!=cfg['data_sha256']: raise ValueError('train data changed')
    with np.load(cfg['data']) as d: original,states,rows=select_anchors(d['train'],d['condition'],cfg)
    if len(rows)!=12: raise ValueError('unexpected anchor count')
    ROOT.mkdir(); RAW.mkdir()
    bank=RAW/'anchors.npz'; np.savez_compressed(bank,original=original,unit_initials=states)
    schedule=[(c,a) for c in range(len(cfg['cases'])) for a in range(len(rows))]
    order=np.random.default_rng(cfg['schedule_seed']).permutation(len(schedule))
    save(ROOT/'manifest.json',dict(config=cfg,config_sha256=digest(CONFIG),anchors=rows,
        bank_sha256=digest(bank),normalization_max_change=float(np.abs(states-original).max()),
        schedule=[schedule[i] for i in order],total_paths=len(schedule)*cfg['paths_per_anchor'],
        code_sha256={p:digest(p) for p in [__file__,'tests/experiments/independent_reference.py','scripts/core/bauer_fast.py']}))


def run():
    if not os.getenv('SLURM_JOB_ID'): raise RuntimeError('CPU Slurm required')
    manifest=json.loads((ROOT/'manifest.json').read_text()); cfg=manifest['config']
    if digest(CONFIG)!=manifest['config_sha256']: raise ValueError('configuration changed')
    if digest(RAW/'anchors.npz')!=manifest['bank_sha256']: raise ValueError('anchor bank changed')
    for p,h in manifest['code_sha256'].items():
        if digest(p)!=h: raise ValueError('source changed')
    dest=ROOT/'summary.json'
    if dest.exists(): raise FileExistsError('no silent retry')
    with np.load(RAW/'anchors.npz') as d: states=d['unit_initials']
    report=dict(status='RUNNING',rows=[],manifest_sha256=digest(ROOT/'manifest.json'),job_id=os.getenv('SLURM_JOB_ID'))
    save(dest,report)
    for ci,ai in manifest['schedule']:
        case=cfg['cases'][ci]; anchor=manifest['anchors'][ai]; n=cfg['paths_per_anchor']
        seeds=cfg['noise_base']+(ci*len(states)+ai)*10000+np.arange(n)
        paths,numerics=generate(np.repeat(states[ai:ai+1],n,axis=0),seeds,
            **case,duration=cfg['duration'],save_dt=case['dt'],anisotropy=cfg['anisotropy'],
            damping=cfg['damping'],theta=anchor['theta'])
        p=RAW/f'case{ci}_anchor{ai}.npz'
        np.savez_compressed(p,spins=paths,seeds=seeds,reduced_time=np.arange(paths.shape[1])*case['dt'])
        report['rows'].append(dict(case=ci,anchor=ai,path=str(p),sha256=digest(p),numerics=numerics))
        save(dest,report)
    report['status']='COMPLETE_FINE_TRANSITION_PILOT_NOT_CERTIFIED'; save(dest,report)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true'); args=p.parse_args()
    prepare() if args.prepare else run()
