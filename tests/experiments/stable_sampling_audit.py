"""Paired-noise integration audit of frozen EMA autonomous window models."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from scripts.workflow.bauer_campaign import save, digest
from tests.experiments.velocity_controls import CartesianPathNet
from tests.experiments.bauer_window import sample_chunks
from tests.experiments.path_observables import path_observables
from tests.experiments.window_learning_analysis import prefix_errors


def paired_errors(a,b):
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('paired finite paths of identical shape required')
    return dict(path_rmse=float(np.sqrt(np.square(a-b).mean())),
        endpoint_mz_rmse=float(np.sqrt(np.square(a[:,-1,:,2].mean(-1)-b[:,-1,:,2].mean(-1)).mean())))


def run(seed,extended=False):
    freeze_path=Path('tests/conf/stable_candidate_freeze.json')
    freeze=json.loads(freeze_path.read_text())
    integration_path=Path('tests/conf/stable_sampling_extended.json') if extended else freeze_path
    integration=json.loads(integration_path.read_text())
    if str(seed) not in freeze['checkpoint_sha256']:raise ValueError('unregistered seed')
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Slurm CUDA allocation required')
    parent=Path(f'tests/results/stable_window_learning/window11/{seed}/summary.json')
    report=json.loads(parent.read_text());cfg=report['config'];base=report['base_config']
    if report['status']!='COMPLETE_EXPLORATORY_NOT_CERTIFIED':raise ValueError('unfinished training')
    checkpoint=Path(cfg['output'])/str(seed)/'step_40000.pt'
    if digest(checkpoint)!=freeze['checkpoint_sha256'][str(seed)]:raise ValueError('checkpoint changed')
    state=torch.load(checkpoint,map_location='cuda',weights_only=False)
    if state['inference_weights']!='ema':raise ValueError('EMA required')
    net=CartesianPathNet().cuda().eval();net.load_state_dict(state['model'])
    experiment='stable_sampling_extended' if extended else 'stable_sampling_audit'
    destination=Path(f'tests/results/{experiment}/{seed}')
    raw=Path(f'tests/data/{experiment}/{seed}')
    if destination.exists() or raw.exists():raise FileExistsError('preserve previous audit')
    with np.load(cfg['data']) as d:
        development=d['development'];conditions=d['development_condition']
    evaluations=[r for r in report['evaluation'] if r['step']==40000]
    if len(evaluations)!=2 or {r['theta'] for r in evaluations}!={.11,.13}:
        raise ValueError('incomplete evaluation')
    destination.mkdir(parents=True);raw.mkdir(parents=True)
    result=dict(status='RUNNING',seed=seed,scope=freeze['scope'],freeze_sha256=digest(freeze_path),
        checkpoint_sha256=digest(checkpoint),data_sha256=digest(cfg['data']),integration_sha256=digest(integration_path),
        code_sha256={p:digest(p) for p in [__file__,'tests/experiments/bauer_window.py',
            'tests/experiments/velocity_controls.py','scripts/model/gate_f.py']},
        job_id=os.getenv('SLURM_JOB_ID'),rows=[],pairs=[])
    save(destination/'summary.json',result)
    for evaluation in evaluations:
        theta=evaluation['theta'];pool=development[np.isclose(conditions[:,0],theta)]
        x=torch.as_tensor(np.repeat(pool[evaluation['initial_indices'],0],cfg['latents'],axis=0)[:,None,:,None],device='cuda',dtype=torch.float32)
        condition=x.new_tensor([theta,.1]).expand(len(x),-1)
        time=torch.linspace(0,1,base['frames'],device='cuda')[:cfg['window_frames']][None].expand(len(x),-1)
        previous=None;previous_steps=None
        for steps in integration['steps']:
            pred=sample_chunks(net,x,condition,time,None,None,
                torch.Generator(device='cuda').manual_seed(freeze['latent_seed']),chunks=20,steps=steps)
            a=pred[:,:,0,:,0].cpu().numpy()
            if not torch.equal(pred[:,0],x):raise ValueError('initial changed')
            np.save(raw/f'T{theta}_steps_{steps}.npy',a)
            row=dict(theta=theta,steps=steps,generated_sha256=digest(raw/f'T{theta}_steps_{steps}.npy'),families=[])
            if steps==128:
                old=np.load(Path(cfg['raw'])/str(seed)/f'step_40000_T{theta}.npy')
                row['previous_evaluation_max_abs_difference']=float(np.abs(a-old).max())
                if row['previous_evaluation_max_abs_difference']>1e-6:raise ValueError('128-step replay changed')
            for family,indices in enumerate(evaluation['reference_indices']):
                reference=pool[indices];prediction=a[family*cfg['latents']:(family+1)*cfg['latents']]
                ref=path_observables(reference);obs=path_observables(prediction)
                np.savez_compressed(destination/f'T{theta}_steps_{steps}_family_{family}.npz',
                    **{f'reference_{k}':v for k,v in ref.items()},**{f'generated_{k}':v for k,v in obs.items()})
                row['families'].append(dict(family=family,reference_indices=indices,
                    prefixes=prefix_errors(obs['magnetization_mean'],ref['magnetization_mean'],base['save_dt']),
                    curve_rmse={k:float(np.sqrt(np.square(obs[k]-ref[k]).mean())) for k in ref if k!='magnetization_per_path'}))
            result['rows'].append(row)
            if previous is not None:
                result['pairs'].append(dict(theta=theta,steps=[previous_steps,steps],**paired_errors(a,previous)))
            previous=a;previous_steps=steps
            save(destination/'summary.json',result)
    result['status']='COMPLETE_SAMPLING_DIAGNOSTIC';save(destination/'summary.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--extended',action='store_true')
    args=p.parse_args();torch.set_num_threads(1);run(args.seed,args.extended)
