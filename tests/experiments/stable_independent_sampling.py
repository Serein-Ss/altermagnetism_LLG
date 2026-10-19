"""Generate the registered G ensembles without accessing reference futures."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from scripts.workflow.bauer_campaign import save,digest
from tests.experiments.velocity_controls import CartesianPathNet
from tests.experiments.bauer_window import sample_chunks


def sample_family(net,initial,theta,cfg,family,batch_size,device):
    n=cfg['n_per_half']
    if batch_size<1 or n%batch_size:raise ValueError('fixed equal batches required')
    paths=[];seeds=[]
    for batch in range(n//batch_size):
        seed=cfg['generated_latent_base']+10000*family+batch
        x=torch.as_tensor(np.repeat(initial[None],batch_size,axis=0)[:,None,:,None],dtype=torch.float32,device=device)
        condition=x.new_tensor([theta,cfg['damping']]).expand(len(x),-1)
        t=torch.linspace(0,1,201,device=device)[:11][None].expand(len(x),-1)
        pred=sample_chunks(net,x,condition,t,None,None,torch.Generator(device=device).manual_seed(seed),
            chunks=20,steps=cfg['generated_sampling_steps'])
        if not torch.equal(pred[:,0],x) or not torch.isfinite(pred).all():raise RuntimeError('invalid generated trajectory')
        if float((pred.norm(dim=-1)-1).abs().max())>1e-5:raise RuntimeError('unit sphere violation')
        paths.append(pred[:,:,0,:,0].cpu().numpy());seeds.append(seed)
    return np.concatenate(paths),seeds


def run(seed):
    sampling_path=Path('tests/conf/stable_independent_sampling.json')
    sampling=json.loads(sampling_path.read_text())
    if seed not in sampling['training_seeds']:raise ValueError('unregistered trained seed')
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm CUDA required')
    manifest_path=Path(sampling['reference_manifest']);manifest=json.loads(manifest_path.read_text());cfg=manifest['config']
    if digest('tests/conf/stable_independent_reference.json')!=manifest['config_sha256']:
        raise ValueError('reference configuration changed')
    if digest(cfg['initials'])!=cfg['initial_sha256'] or digest(cfg['model_freeze'])!=manifest['model_freeze_sha256']:
        raise ValueError('initials or candidate freeze changed')
    if cfg['duration']!=16000 or cfg['save_dt']!=80:raise ValueError('unsupported time contract')
    frozen=json.loads(Path(cfg['model_freeze']).read_text())
    checkpoint=Path(f'tests/output/stable_window_learning/window11/{seed}/step_40000.pt')
    if digest(checkpoint)!=frozen['checkpoint_sha256'][str(seed)]:raise ValueError('checkpoint changed')
    state=torch.load(checkpoint,map_location='cuda',weights_only=False)
    if state['inference_weights']!='ema':raise ValueError('EMA inference required')
    net=CartesianPathNet().cuda().eval();net.load_state_dict(state['model'])
    initials=np.load(cfg['initials'])
    destination=Path(f'tests/results/stable_independent_sampling/{seed}')
    raw=Path(f'tests/data/stable_independent_sampling/{seed}')
    if destination.exists() or raw.exists():raise FileExistsError('preserve previous generation')
    destination.mkdir(parents=True);raw.mkdir(parents=True)
    result=dict(status='RUNNING',training_seed=seed,checkpoint_sha256=digest(checkpoint),
        reference_manifest_sha256=digest(manifest_path),sampling_config_sha256=digest(sampling_path),
        scope=sampling['scope'],latent_mapping=sampling['latent_mapping'],batch_size=sampling['batch_size'],
        job_id=os.getenv('SLURM_JOB_ID'),code_sha256={p:digest(p) for p in [__file__,
            'tests/experiments/bauer_window.py','tests/experiments/velocity_controls.py','scripts/model/gate_f.py']},rows=[])
    save(destination/'summary.json',result)
    for family,initial in enumerate(initials):
        for theta in cfg['theta']:
            torch.cuda.synchronize();begin=time.monotonic()
            paths,latent_seeds=sample_family(net,initial,theta,cfg,family,sampling['batch_size'],'cuda')
            torch.cuda.synchronize();seconds=time.monotonic()-begin
            path=raw/f'family_{family}_T{theta}.npz'
            np.savez_compressed(path,spins=paths,initial=initial.astype('f4'),
                batch_seeds=np.asarray(latent_seeds),reduced_time=np.arange(201)*cfg['save_dt'])
            result['rows'].append(dict(family=family,theta=theta,paths=len(paths),sampling_steps=cfg['generated_sampling_steps'],
                batch_seeds=latent_seeds,generation_seconds=seconds,seconds_per_path=seconds/len(paths),data_sha256=digest(path)))
            save(destination/'summary.json',result)
    result['status']='COMPLETE_GENERATED_ENSEMBLE_NOT_CERTIFIED';save(destination/'summary.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--device',choices=['cuda'],required=True)
    args=p.parse_args();torch.set_num_threads(1);run(args.seed)
