"""Paired-source integration checks on frozen experimental checkpoints."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from scripts.model.gate_f import reference_path, sample
from scripts.workflow.bauer_campaign import save, digest
from tests.experiments.velocity_controls import CartesianPathNet
from tests.experiments.path_observables import path_observables
from tests.experiments.bauer_checkpoint_diagnostics import statistics


def run(seed):
    if seed not in (11,22,33):raise ValueError('unregistered training seed')
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Slurm CUDA allocation required')
    for experiment in ('cartesian_overfit','temporal_dilation_overfit'):
        report_path=Path(f'tests/results/{experiment}/mixed_eight/{seed}/summary.json')
        report=json.loads(report_path.read_text());cfg=report['config'];base=report['base_config']
        if report['status']!='COMPLETE_EXPLORATORY_NOT_CERTIFIED':raise ValueError('unfinished checkpoint run')
        destination=Path(f'tests/results/cartesian_sampling_audit/{experiment}/{seed}')
        raw=Path(f'tests/data/cartesian_sampling_audit/{experiment}/{seed}')
        if destination.exists() or raw.exists():raise FileExistsError('preserve previous audit')
        checkpoint=Path(cfg['output'])/str(seed)/'step_6000.pt'
        net=CartesianPathNet(temporal_dilations=cfg.get('temporal_dilations')).cuda().eval()
        net.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['model'])
        with np.load(cfg['data']) as d:reference=d['train'][cfg['train_indices']]
        initial=torch.as_tensor(np.repeat(reference[:1,0],cfg['latents'],axis=0)[:,None,:,None],device='cuda')
        condition=initial.new_tensor([cfg['evaluation_theta'][0],.1]).expand(len(initial),-1)
        t=torch.linspace(0,1,base['frames'],device='cuda')[None].expand(len(initial),-1)
        source=torch.cat([reference_path(initial[j:j+1],base['frames'],
            torch.Generator(device='cuda').manual_seed(913000+j)) for j in range(len(initial))])
        raw.mkdir(parents=True);destination.mkdir(parents=True)
        np.save(raw/'source.npy',source.cpu().numpy())
        result=dict(status='RUNNING',checkpoint_sha256=digest(checkpoint),
            source_sha256=digest(raw/'source.npy'),training_seed=seed,
            code_sha256={p:digest(p) for p in [__file__,'scripts/model/gate_f.py','tests/experiments/velocity_controls.py']},
            job_id=os.getenv('SLURM_JOB_ID'),scope='Selected training-family integration audit only; no physical certification. Legacy threshold/dwell are descriptive only.',
            reference=statistics(reference,base['save_dt'],cfg['threshold'],cfg['dwell']),rows=[],pairs=[])
        previous=None; previous_steps=None
        for steps in (32,64,128,256):
            pred=sample(net,initial,condition,t,None,None,torch.Generator(device='cuda').manual_seed(913000),steps=steps,source=source)
            if not torch.equal(pred[:,0],initial):raise ValueError('initial changed')
            a=pred[:,:,0,:,0].cpu().numpy();observables=path_observables(a)
            np.save(raw/f'steps_{steps}.npy',a)
            np.savez_compressed(destination/f'profiles_{steps}.npz',**observables)
            row=dict(steps=steps,statistics=statistics(a,base['save_dt'],cfg['threshold'],cfg['dwell']))
            if steps==128:
                old=np.load(Path(cfg['raw'])/str(seed)/f"step_6000_T{cfg['evaluation_theta'][0]}.npy")
                row['previous_evaluation_max_abs_difference']=float(np.abs(a-old).max())
            result['rows'].append(row)
            if previous is not None:
                result['pairs'].append(dict(steps=[previous_steps,steps],
                    path_rmse=float(np.sqrt(np.square(a-previous).mean())),
                    endpoint_mz_rmse=float(np.sqrt(np.square(a[:,-1,:,2].mean(-1)-previous[:,-1,:,2].mean(-1)).mean()))))
            previous=a;previous_steps=steps
            save(destination/'summary.json',result)
        result['status']='COMPLETE_SAMPLING_DIAGNOSTIC';save(destination/'summary.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    args=p.parse_args();torch.set_num_threads(1);run(args.seed)
