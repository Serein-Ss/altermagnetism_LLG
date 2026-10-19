"""Frozen conditional one-step diagnostic on preselected true LLG states."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest, save

CONFIG = Path('tests/conf/conditional_restart.json')
ROOT = Path('tests/results/conditional_restart')
RAW = Path('tests/data/conditional_restart')


def selections(cfg, paths):
    rng = np.random.default_rng(cfg['selection_seed'])
    return [[(0,0)]+[(int(p),f) for p in rng.choice(paths,cfg['parent_paths'],replace=False) for f in cfg['frames']] for _ in range(12)]


def noise_seeds(cfg, case, state, count):
    return cfg['noise_base']+(case*count+state)*10000+np.arange(cfg['paths'])


def prepare():
    cfg = json.loads(CONFIG.read_text())
    rp = Path('tests/results/fresh_rollout_reference/summary.json')
    ref = json.loads(rp.read_text())
    mp = rp.with_name('manifest.json')
    manifest = json.loads(mp.read_text())
    assert ref['status'] == 'COMPLETE_FRESH_ROLLOUT_REFERENCE_NOT_CERTIFIED'
    assert digest(mp) == ref['manifest_sha256']
    for model in manifest['frozen_models']:
        assert digest(model['checkpoint']) == model['sha256']
        assert digest(model['report']) == model['report_sha256']
    states,rows = [],[]
    for anchor,selected in enumerate(selections(cfg,256)):
        source = next(r for r in ref['rows'] if r['anchor']==anchor and r['case']==1)
        assert digest(source['path']) == source['sha256']
        with np.load(source['path']) as z:
            for parent,frame in selected:
                states.append(z['spins'][parent,frame].copy())
                rows.append(dict(anchor=anchor,parent_path=parent,parent_noise_seed=int(z['seeds'][parent]),
                    frame=frame,parent_time=float(z['reduced_time'][frame]),source_sha256=source['sha256'],
                    state=manifest['anchors'][anchor]))
    states = np.asarray(states)
    assert states.shape == (60,25,3) and np.isfinite(states).all()
    assert np.max(abs(np.linalg.norm(states,axis=-1)-1)) < 1e-10
    if ROOT.exists() or RAW.exists():
        raise FileExistsError('preserve prior diagnostic')
    ROOT.mkdir(); RAW.mkdir()
    np.savez_compressed(RAW/'states.npz',unit_initials=states)
    schedule = [(c,s) for s in range(60) for c in range(3)]
    rng = np.random.default_rng(cfg['schedule_seed'])
    save(ROOT/'manifest.json',dict(config=cfg,config_sha256=digest(CONFIG),rows=rows,
        physics=manifest['config'],frozen_models=manifest['frozen_models'],reference_sha256=digest(rp),
        bank_sha256=digest(RAW/'states.npz'),schedule=[schedule[i] for i in rng.permutation(180)],
        model_order=rng.permutation(60).tolist(),
        code_sha256={p:digest(p) for p in [__file__,'tests/experiments/independent_reference.py',
            'scripts/core/bauer_fast.py','tests/experiments/director_midpoint_sampling.py',
            'tests/experiments/projected_midpoint_sampler.py']}))


def inputs():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    assert digest(CONFIG) == manifest['config_sha256']
    assert digest(RAW/'states.npz') == manifest['bank_sha256']
    for p,h in manifest['code_sha256'].items():
        assert digest(p) == h
    with np.load(RAW/'states.npz') as z:
        states = z['unit_initials']
    return manifest,states


def run_reference():
    from tests.experiments.independent_reference import generate
    assert os.getenv('SLURM_JOB_ID'), 'CPU Slurm required'
    m,states = inputs(); cfg=m['config']
    dest=ROOT/'reference.json'
    if dest.exists(): raise FileExistsError(dest)
    report=dict(status='RUNNING',rows=[],manifest_sha256=digest(ROOT/'manifest.json'),job_id=os.environ['SLURM_JOB_ID'])
    for case,state in m['schedule']:
        seeds=noise_seeds(cfg,case,state,len(states))
        p,numerics=generate(np.repeat(states[state:state+1],cfg['paths'],axis=0),seeds,
            **cfg['cases'][case],duration=cfg['lag'],save_dt=cfg['lag'],
            anisotropy=m['physics']['anisotropy'],damping=m['physics']['damping'],theta=m['rows'][state]['state']['theta'])
        path=RAW/f'reference_case{case}_state{state}.npz'
        np.savez_compressed(path,spins=p,seeds=seeds,reduced_time=[0,cfg['lag']])
        report['rows'].append(dict(case=case,state_id=state,path=str(path),sha256=digest(path),numerics=numerics))
        save(dest,report)
    report['status']='COMPLETE_CONDITIONAL_REFERENCE_NOT_CERTIFIED'; save(dest,report)


def run_model(seed,device):
    import torch
    from tests.experiments.director_transition import DirectorTransition
    from tests.experiments.director_midpoint_sampling import rollout
    if device!='cuda' or not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Slurm CUDA allocation required')
    m,states=inputs(); cfg=m['config']
    frozen=next(r for r in m['frozen_models'] if r['seed']==seed)
    assert digest(frozen['checkpoint'])==frozen['sha256']
    assert digest(frozen['report'])==frozen['report_sha256']
    for p,h in json.loads(Path(frozen['report']).read_text())['code_sha256'].items():
        assert digest(p)==h
    checkpoint=torch.load(frozen['checkpoint'],map_location='cpu',weights_only=False)
    training=dict(checkpoint['config'])
    assert training['lag']==cfg['lag']
    training.update(rollout_steps=1,generated_paths=cfg['paths'],generation_batch=cfg['generation_batch'])
    model=DirectorTransition(checkpoint['theta_mean'],checkpoint['theta_std'],training['width']).to(device)
    model.load_state_dict(checkpoint['ema'],strict=True); model.eval()
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    dest=ROOT/f'model{seed}.json'
    if dest.exists(): raise FileExistsError(dest)
    report=dict(status='RUNNING',rows=[],manifest_sha256=digest(ROOT/'manifest.json'),
        checkpoint_sha256=frozen['sha256'],job_id=os.environ['SLURM_JOB_ID'],config=training)
    for state in m['model_order']:
        latent=cfg['latent_base']+seed*1000+state
        p,rng=rollout(model,states[state],m['rows'][state]['state']['theta'],training,
            checkpoint['latent_scale'],cfg['flow_steps'],latent,device)
        assert p.shape==(cfg['paths'],2,25,3) and np.isfinite(p).all()
        assert np.max(abs(np.linalg.norm(p,axis=-1)-1))<1e-5
        path=RAW/f'model{seed}_state{state}.npz'
        np.savez_compressed(path,spins=p,rng_state=rng,reduced_time=[0,cfg['lag']])
        report['rows'].append(dict(state_id=state,path=str(path),sha256=digest(path),latent_seed=latent))
        save(dest,report)
    report['status']='COMPLETE_CONDITIONAL_MODEL_NOT_CERTIFIED'; save(dest,report)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','reference','model'])
    p.add_argument('--seed',type=int,choices=[11,22,33]); p.add_argument('--device')
    args=p.parse_args()
    if args.mode=='prepare': prepare()
    elif args.mode=='reference': run_reference()
    else: run_model(args.seed,args.device)
