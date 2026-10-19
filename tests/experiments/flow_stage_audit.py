"""Frozen-model flow-time diagnostics; empirical source-point bias decomposition."""
import argparse
import os
import json
from pathlib import Path
import torch
import numpy as np
from scripts.model.gate_f import reference_path, path_interpolate
from scripts.workflow.bauer_campaign import save, digest
from tests.experiments.velocity_controls import CartesianPathNet


def source_error_decomposition(prediction, targets):
    """targets[K,S,...], prediction[S,...]; targets enumerate equal-weight atoms."""
    mean=targets.mean(0)
    return dict(total_mse=float((targets-prediction[None]).square().mean()),
                target_variance=float((targets-mean[None]).square().mean()),
                model_bias_mse=float((prediction-mean).square().mean()),
                mean_target_square=float(mean.square().mean()))


@torch.no_grad()
def run(seed):
    if seed not in (11,22,33):raise ValueError('unknown seed')
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm CUDA required')
    dest=Path(f'tests/results/flow_stage_audit/{seed}')
    if dest.exists():raise FileExistsError(dest)
    rows=[]
    for experiment in ('cartesian_overfit','temporal_dilation_overfit','global_context_overfit'):
        report=json.loads(Path(f'tests/results/{experiment}/mixed_eight/{seed}/summary.json').read_text())
        if report['status']!='COMPLETE_EXPLORATORY_NOT_CERTIFIED':raise ValueError('unfinished run')
        cfg=report['config']
        with np.load(cfg['data']) as d:pool=d['train'][cfg['train_indices']]
        y=torch.as_tensor(pool[:,:,None,:,None],device='cuda').repeat_interleave(8,dim=0)
        gen=torch.Generator(device='cuda').manual_seed(920000)
        source=reference_path(y[:8,0],y.shape[1],gen).repeat(8,1,1,1,1,1)
        t=torch.linspace(0,1,y.shape[1],device='cuda')[None].expand(len(y),-1)
        c=y.new_tensor([cfg['evaluation_theta'][0],.1]).expand(len(y),-1)
        net=CartesianPathNet(temporal_dilations=cfg.get('temporal_dilations'),global_context=cfg.get('global_context',False)).cuda().eval()
        for step in (600,3000,6000):
            checkpoint=Path(cfg['output'])/str(seed)/f'step_{step}.pt'
            net.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['model'])
            for tau in (0.,.01,.05,.1,.25,.5,.75,.9,.99):
                flow=y.new_full((len(y),),tau)
                x,target,diagnostics=path_interpolate(source,y,flow,torch.Generator(device='cuda').manual_seed(920001))
                pred=net(x,flow,y[:,0],c,t,None,None)
                error=(pred[:,1:]-target[:,1:]).square().flatten(1).mean(1)
                if not torch.isfinite(error).all():raise ValueError('nonfinite stage error')
                row=dict(model=experiment,step=step,tau=tau,checkpoint_sha256=digest(checkpoint),
                    mse=float(error.mean()),target_square=float(target[:,1:].square().mean()),
                    per_target_source_mse=error.reshape(8,8).cpu().tolist(),interpolation=diagnostics)
                if tau==0:
                    p=pred.reshape(8,8,*pred.shape[1:])[:,:,1:]
                    if not torch.equal(p,p[:1].expand_as(p)):raise ValueError('target identity leaked at source')
                    row['source_decomposition']=source_error_decomposition(p[0],target.reshape(8,8,*target.shape[1:])[:,:,1:])
                rows.append(row)
    save(dest/'summary.json',dict(status='COMPLETE_FLOW_STAGE_DIAGNOSTIC',seed=seed,
        scope='Eight selected training targets and eight paired sources, not independent physical validation. Source decomposition is exact over equal-weight empirical targets conditional on each tested source, not a general irreducible-loss claim for tau>0. No training or selection.',
        code_sha256={p:digest(p) for p in [__file__,'scripts/model/gate_f.py','tests/experiments/velocity_controls.py']},rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    args=p.parse_args();torch.set_num_threads(1);run(args.seed)
