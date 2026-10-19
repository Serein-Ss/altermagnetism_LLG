"""Paired development trajectories from all controlled arms, no model selection."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from scripts.workflow.bauer_campaign import save, digest
from tests.experiments.bauer_checkpoint_diagnostics import initial_families
from tests.experiments.stable_independent_sampling import sample_family
from tests.experiments.velocity_controls import CartesianPathNet
from tests.experiments.flow_film_model import FlowFiLMPathNet


def development_groups(paths, conditions, theta):
    indices=np.flatnonzero(np.isclose(conditions[:,0],theta))
    pool=paths[indices]
    first,groups=initial_families(pool,2,16)
    if len(pool)!=32 or len(np.unique(groups))!=32:
        raise ValueError('expected exactly two complete development families')
    return pool[first,0],indices[groups]


def run(seed):
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available(): raise RuntimeError('Slurm CUDA required')
    path=Path('tests/conf/flow_film_evaluation.json'); cfg=json.loads(path.read_text())
    train=json.loads(Path(cfg['training_config']).read_text())
    if seed not in cfg['seeds']: raise ValueError('unregistered seed')
    if digest(train['data'])!=train['data_sha256']: raise ValueError('data changed')
    checkpoints={}
    # Check every training arm/seed before generating any evaluation samples.
    for arm in train['arms']:
        for s in train['seeds']:
            p=Path(f'tests/results/flow_film_controlled/{arm}/{s}/summary.json')
            r=json.loads(p.read_text())
            if r['status']!='COMPLETE_TRAINING_NOT_EVALUATED' or r['config_sha256']!=digest(cfg['training_config']):
                raise ValueError('complete matching training batch required')
            if digest(r['checkpoint'])!=r['checkpoint_sha256']: raise ValueError('trained checkpoint changed')
            if s==seed: checkpoints[arm]=(Path(r['checkpoint']),r['checkpoint_sha256'])
    freeze=json.loads(Path(train['freeze']).read_text())
    frozen=Path(f'tests/output/stable_window_learning/window11/{seed}/step_40000.pt')
    if digest(frozen)!=freeze['checkpoint_sha256'][str(seed)]: raise ValueError('frozen checkpoint changed')
    checkpoints['frozen']=(frozen,digest(frozen))
    with np.load(train['data']) as d:
        dev=d['development']; conditions=d['development_condition']
    groups={theta:development_groups(dev,conditions,theta) for theta in cfg['theta']}
    for arm in cfg['arms']:
        out=Path(f'tests/results/flow_film_evaluation/{arm}/{seed}')
        raw=Path(f'tests/data/flow_film_evaluation/{arm}/{seed}')
        if out.exists() or raw.exists(): raise FileExistsError('preserve previous evaluation')
        out.mkdir(parents=True); raw.mkdir(parents=True)
        cp,sha=checkpoints[arm]; state=torch.load(cp,map_location='cuda',weights_only=False)
        if state['inference_weights']!='ema': raise ValueError('EMA required')
        net=(FlowFiLMPathNet if arm=='flow_film' else CartesianPathNet)().cuda().eval()
        net.load_state_dict(state['model'])
        result=dict(status='RUNNING',arm=arm,seed=seed,config=cfg,config_sha256=digest(path),
            data_sha256=train['data_sha256'],checkpoint_sha256=sha,rows=[],
            code_sha256={p:digest(p) for p in [__file__,'tests/experiments/stable_independent_sampling.py',
                'tests/experiments/flow_film_model.py','tests/experiments/velocity_controls.py',
                'tests/experiments/bauer_window.py','scripts/model/gate_f.py']})
        save(out/'summary.json',result)
        for theta,(initials,indices) in groups.items():
            for family,initial in enumerate(initials):
                torch.cuda.synchronize(); begin=time.monotonic()
                pred,latent_seeds=sample_family(net,initial,theta,cfg,family,cfg['batch_size'],'cuda')
                torch.cuda.synchronize(); elapsed=time.monotonic()-begin
                dest=raw/f'family{family}_T{theta}.npz'
                np.savez_compressed(dest,spins=pred,reduced_time=np.arange(201)*80,
                    reference_indices=indices[family],batch_seeds=latent_seeds)
                result['rows'].append(dict(family=family,theta=theta,path=str(dest),sha256=digest(dest),
                    reference_indices=indices[family].tolist(),generation_seconds=elapsed,seconds_per_path=elapsed/len(pred)))
                save(out/'summary.json',result)
        result['status']='COMPLETE_DEVELOPMENT_GENERATION_NOT_CERTIFIED'; save(out/'summary.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True)
    p.add_argument('--device',choices=['cuda'],required=True)
    args=p.parse_args(); torch.set_num_threads(1); run(args.seed)
