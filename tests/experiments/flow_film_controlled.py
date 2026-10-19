"""Matched additional-update control versus flow-time FiLM, from frozen EMA."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from scripts.workflow.bauer_campaign import save, digest
from scripts.model.gate_f import matching_loss
from tests.experiments.bauer_window import crop_training_windows
from tests.experiments.velocity_controls import CartesianPathNet
from tests.experiments.flow_film_model import FlowFiLMPathNet
from tests.experiments.stable_optimization import cosine_lr, update_ema


def warm_start(arm, weights, device='cpu', width=64, blocks=4):
    if arm not in ('control','flow_film'): raise ValueError('unknown arm')
    net=(CartesianPathNet if arm=='control' else FlowFiLMPathNet)(width=width,blocks=blocks).to(device)
    missing,unexpected=net.load_state_dict(weights,strict=False)
    expected={f'flow_film.{i}.{k}' for i in range(blocks) for k in ('weight','bias')} if arm=='flow_film' else set()
    if set(missing)!=expected or unexpected: raise ValueError('incomplete backbone warm start')
    return net


def run(seed, arm):
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available(): raise RuntimeError('Slurm CUDA required')
    config=Path('tests/conf/flow_film_controlled.json'); cfg=json.loads(config.read_text())
    if seed not in cfg['seeds'] or arm not in cfg['arms']: raise ValueError('unregistered arm/seed')
    if digest(cfg['data'])!=cfg['data_sha256']: raise ValueError('training data changed')
    cp=Path(f'tests/output/stable_window_learning/window11/{seed}/step_40000.pt')
    freeze=json.loads(Path(cfg['freeze']).read_text())
    if digest(cp)!=freeze['checkpoint_sha256'][str(seed)]: raise ValueError('source checkpoint changed')
    state=torch.load(cp,map_location='cuda',weights_only=False)
    if state['inference_weights']!='ema': raise ValueError('EMA source required')
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    net=warm_start(arm,state['model'],'cuda'); net.train()
    optimizer=torch.optim.Adam(net.parameters(),lr=cfg['learning_rate'])
    ema={k:v.detach().clone() for k,v in net.state_dict().items()}
    with np.load(cfg['data']) as d:
        y=torch.as_tensor(d['train'][:,:,None,:,None],dtype=torch.float32,device='cuda')
        c=torch.as_tensor(d['condition'],dtype=torch.float32,device='cuda')
    out=Path(f'tests/output/flow_film_controlled/{arm}/{seed}')
    dest=Path(f'tests/results/flow_film_controlled/{arm}/{seed}')
    if out.exists() or dest.exists(): raise FileExistsError('preserve existing training')
    out.mkdir(parents=True); dest.mkdir(parents=True)
    report=dict(status='TRAINING',config=cfg,config_sha256=digest(config),seed=seed,arm=arm,
        source_checkpoint_sha256=digest(cp),parameter_count=sum(p.numel() for p in net.parameters()),
        job_id=os.getenv('SLURM_JOB_ID'),code_sha256={p:digest(p) for p in [__file__,
            'tests/experiments/flow_film_model.py','tests/experiments/velocity_controls.py',
            'tests/experiments/bauer_window.py','tests/experiments/stable_optimization.py','scripts/model/gate_f.py']},milestones=[])
    save(dest/'summary.json',report)
    generator=torch.Generator(device='cuda').manual_seed(cfg['random_seed']+seed)
    t=torch.linspace(0,1,y.shape[1],device='cuda')[:cfg['window_frames']][None].expand(cfg['batch_size'],-1)
    losses=[]; begin=time.monotonic()
    for step in range(1,cfg['steps']+1):
        for group in optimizer.param_groups:
            group['lr']=cosine_lr(step,cfg['steps'],cfg['learning_rate'],cfg['min_lr'])
        indices=torch.randint(len(y),(cfg['batch_size'],),device='cuda',generator=generator)
        batch=crop_training_windows(y[indices],cfg['window_frames'],generator)
        optimizer.zero_grad(set_to_none=True)
        loss,_=matching_loss(net,batch,c[indices],t,None,None,generator)
        if not torch.isfinite(loss): raise ValueError('nonfinite training loss')
        loss.backward(); grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        if not torch.isfinite(grad): raise ValueError('nonfinite gradient')
        optimizer.step(); update_ema(ema,net.state_dict(),cfg['ema_decay']); losses.append(float(loss))
        if step%500==0:
            row=dict(step=step,last500_mean=float(np.mean(losses[-500:])),wall_seconds=time.monotonic()-begin)
            report['milestones'].append(row); save(dest/'summary.json',report); print(json.dumps(row),flush=True)
    checkpoint=out/'step_6000.pt'
    torch.save(dict(model=ema,raw_model=net.state_dict(),inference_weights='ema',optimizer_weights='raw_model',
        optimizer=optimizer.state_dict(),generator_state=generator.get_state(),step=cfg['steps'],arm=arm,seed=seed,
        experiment=cfg,source_checkpoint_sha256=digest(cp)),checkpoint)
    save(dest/'training_loss.json',losses)
    report.update(status='COMPLETE_TRAINING_NOT_EVALUATED',checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint))
    save(dest/'summary.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True)
    p.add_argument('--arm',choices=['control','flow_film'],required=True)
    p.add_argument('--device',choices=['cuda'],required=True)
    args=p.parse_args(); torch.set_num_threads(1); run(args.seed,args.arm)
