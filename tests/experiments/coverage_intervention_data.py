"""Independent short-trajectory bank for fixed-count coverage intervention."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest,save
from tests.experiments.independent_reference import generate

CONFIG=Path('tests/conf/coverage_intervention.json')
ROOT=Path('tests/results/coverage_intervention_data')
RAW=Path('tests/data/coverage_intervention')


def initial_bank(cfg):
    rng=np.random.default_rng(cfg['initial_seed'])
    states=[]; rows=[]
    for kind in cfg['kinds']:
        for family in range(cfg['families']):
            n=cfg['sites']; phi=rng.uniform(0,2*np.pi)
            if kind=='aligned':
                base=np.zeros((n,3)); base[:,2]=1 if family%2==0 else -1
            elif kind=='tilted':
                angle=rng.uniform(*cfg['tilt_range'])
                base=np.tile([np.sin(angle)*np.cos(phi),np.sin(angle)*np.sin(phi),np.cos(angle)],(n,1))
            elif kind=='wall':
                width=rng.uniform(*cfg['wall_width_range']); center=rng.uniform(*cfg['wall_center_range'])
                z=(np.arange(n)-center)/width
                base=np.stack([np.cos(phi)/np.cosh(z),np.sin(phi)/np.cosh(z),np.tanh(z)],axis=-1)
                if family%2: base=-base
            else: raise ValueError('unknown initial kind')
            sigma=cfg['aligned_noise'] if kind=='aligned' else cfg['texture_noise']
            x=base+sigma*rng.normal(size=(n,3)); x/=np.linalg.norm(x,axis=-1,keepdims=True)
            states.append(x); rows.append(dict(kind=kind,family=family,initial_id=len(states)-1))
    return np.asarray(states),rows


def prepare():
    cfg=json.loads(CONFIG.read_text())
    states,rows=initial_bank(cfg)
    forbidden=[]
    for p in ['tests/data/fine_transition_pilot/anchors.npz','tests/data/fresh_rollout_reference/anchors.npz','tests/data/conditional_restart/states.npz']:
        with np.load(p) as z: forbidden.extend(z['unit_initials'])
    assert all(not np.array_equal(a,b) for a in states for b in forbidden)
    assert states.shape==(12,25,3) and np.isfinite(states).all()
    assert np.max(abs(np.linalg.norm(states,axis=-1)-1))<1e-12
    models=[]
    for seed in cfg['seeds']:
        p=Path(f'tests/results/director_budget/24000/{seed}/summary.json')
        r=json.loads(p.read_text()); assert r['status']=='COMPLETE_TRAINING_AND_GENERATION_NOT_CERTIFIED'
        assert digest(r['checkpoint'])==r['checkpoint_sha256']
        models.append(dict(seed=seed,report=str(p),report_sha256=digest(p),checkpoint=r['checkpoint'],
            checkpoint_sha256=r['checkpoint_sha256'],latent_scale=r['latent_scale']))
    assert len({r['latent_scale'] for r in models})==1
    if ROOT.exists() or RAW.exists(): raise FileExistsError('preserve existing intervention')
    ROOT.mkdir(); RAW.mkdir()
    np.savez_compressed(RAW/'initials.npz',unit_initials=states)
    anchors=[dict(r,theta=t) for t in cfg['theta'] for r in rows]
    save(ROOT/'manifest.json',dict(config=cfg,config_sha256=digest(CONFIG),bank_sha256=digest(RAW/'initials.npz'),
        anchors=anchors,schedule=np.random.default_rng(cfg['schedule_seed']).permutation(24).tolist(),
        frozen_control=models,old_source_sha256=digest('tests/results/fine_transition_pilot/summary.json'),
        code_sha256={p:digest(p) for p in [__file__,'tests/experiments/independent_reference.py','scripts/core/bauer_fast.py']},
        excluded_bank_sha256={p:digest(p) for p in ['tests/data/fine_transition_pilot/anchors.npz','tests/data/fresh_rollout_reference/anchors.npz','tests/data/conditional_restart/states.npz']}))


def run():
    assert os.getenv('SLURM_JOB_ID'),'CPU Slurm required'
    mp=ROOT/'manifest.json'; m=json.loads(mp.read_text()); cfg=m['config']
    assert digest(CONFIG)==m['config_sha256'] and digest(RAW/'initials.npz')==m['bank_sha256']
    for p,h in m['code_sha256'].items(): assert digest(p)==h
    with np.load(RAW/'initials.npz') as z: states=z['unit_initials']
    dest=ROOT/'summary.json'
    if dest.exists(): raise FileExistsError(dest)
    report=dict(status='RUNNING',rows=[],manifest_sha256=digest(mp),job_id=os.environ['SLURM_JOB_ID'])
    for anchor in m['schedule']:
        a=m['anchors'][anchor]; n=cfg['paths_per_anchor']
        seeds=cfg['noise_base']+anchor*10000+np.arange(n)
        initial=np.repeat(states[a['initial_id']:a['initial_id']+1],n,axis=0)
        p,numerics=generate(initial,seeds,**{k:cfg[k] for k in ['dt','duration','save_dt','anisotropy','damping','method']},theta=a['theta'])
        path=RAW/f'anchor{anchor}.npz'
        np.savez_compressed(path,spins=p,seeds=seeds,reduced_time=np.arange(p.shape[1])*cfg['save_dt'])
        report['rows'].append(dict(anchor=anchor,state=a,path=str(path),sha256=digest(path),numerics=numerics))
        save(dest,report)
    report['status']='COMPLETE_COVERAGE_INTERVENTION_DATA_NOT_CERTIFIED'; save(dest,report)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true'); args=p.parse_args()
    prepare() if args.prepare else run()
