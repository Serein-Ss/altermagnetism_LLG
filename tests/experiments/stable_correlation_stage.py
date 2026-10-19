"""Replay supervised stage audit and measure signed correlation-rate residuals."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from scripts.workflow.bauer_campaign import save, digest
from scripts.model.gate_f import reference_path, path_interpolate
from tests.experiments.velocity_controls import CartesianPathNet
from tests.experiments.stable_flow_stage import free_frame_risk
from tests.experiments.correlation_velocity import correlation_rate


@torch.no_grad()
def run():
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Slurm CUDA required')
    source = Path('tests/results/stable_flow_stage/summary.json')
    old = json.loads(source.read_text()); cfg = old['config']
    if old['status'] != 'COMPLETE_SUPERVISED_RISK_NOT_CERTIFIED' or len(old['rows']) != 6:
        raise ValueError('complete risk audit required')
    if digest(cfg['data']) != cfg['data_sha256']: raise ValueError('data changed')
    with np.load(cfg['data']) as d:
        arrays = {k:d[k] for k in ('train','condition','development','development_condition')}
    out = Path('tests/results/stable_correlation_stage')
    if out.exists(): raise FileExistsError('preserve prior audit')
    out.mkdir()
    report = dict(status='RUNNING', source_sha256=digest(source), rows=[],
        scope='Signed correlation derivative residual on supervised interpolation states, NOT generated-state transport or physical-time derivative. No complete path-law or causal certificate.',
        code_sha256={p:digest(p) for p in [__file__, 'tests/experiments/correlation_velocity.py',
                    'tests/experiments/stable_flow_stage.py','tests/experiments/velocity_controls.py','scripts/model/gate_f.py']})
    for row in old['rows']:
        seed,split = row['seed'],row['split']; si = 0 if split=='train' else 1
        if digest(row['path']) != row['sha256']: raise ValueError('risk array changed')
        with np.load(row['path']) as d:
            starts = d['window_starts']; expected = d['squared_vector_error']
        a=arrays[split]; c_all=arrays['condition' if si==0 else 'development_condition']
        n,w=starts.shape; f=cfg['window_frames']; parent=np.repeat(np.arange(n),w)
        windows=a[parent[:,None],starts.ravel()[:,None]+np.arange(f)]
        cp=Path(f'tests/output/stable_window_learning/window11/{seed}/step_40000.pt')
        if digest(cp)!=row['checkpoint_sha256']: raise ValueError('checkpoint changed')
        state=torch.load(cp,map_location='cuda',weights_only=False)
        if state['inference_weights']!='ema': raise ValueError('EMA required')
        net=CartesianPathNet().cuda().eval(); net.load_state_dict(state['model'])
        rates={k:np.empty((n*w,len(cfg['tau']),f-1)) for k in ('predicted','target')}
        replay_max=0.
        for offset in range(0,n*w,cfg['batch_size']):
            stop=min(offset+cfg['batch_size'],n*w)
            y=torch.as_tensor(windows[offset:stop,:,None,:,None],dtype=torch.float32,device='cuda')
            c=torch.as_tensor(c_all[parent[offset:stop]],dtype=y.dtype,device='cuda')
            t=torch.linspace(0,1,a.shape[1],device='cuda')[:f][None].expand(len(y),-1)
            local_seed=cfg['random_seed']+10000+si*100000+offset
            z=reference_path(y[:,0],f,torch.Generator(device='cuda').manual_seed(local_seed))
            for ti,tau in enumerate(cfg['tau']):
                flow=y.new_full((len(y),),tau)
                x,target,_=path_interpolate(z,y,flow,torch.Generator(device='cuda').manual_seed(local_seed+1))
                pred=net(x,flow,y[:,0],c,t,None,None)
                err,_=free_frame_risk(pred,target); err=err.cpu().numpy()
                prior=expected.reshape(n*w,len(cfg['tau']),f-1)[offset:stop,ti]
                np.testing.assert_allclose(err,prior,rtol=1e-5,atol=1e-6)
                replay_max=max(replay_max,float(np.abs(err-prior).max()))
                for name,v in (('predicted',pred),('target',target)):
                    rates[name][offset:stop,ti]=correlation_rate(x[:,:,0,:,0],v[:,:,0,:,0])[:,1:].cpu().numpy()
        rates={k:v.reshape(n,w,len(cfg['tau']),f-1) for k,v in rates.items()}
        residual=rates['predicted']-rates['target']
        path=out/f'{split}_seed{seed}.npz'
        np.savez_compressed(path,**rates,residual=residual,window_starts=starts,
                            parent_index=np.arange(n),condition=c_all,tau=cfg['tau'])
        report['rows'].append(dict(seed=seed,split=split,path=str(path),sha256=digest(path),
            replay_max_abs_error=replay_max,
            mean_residual_by_theta={str(theta):residual[np.isclose(c_all[:,0],theta)].mean(axis=(0,1)).tolist()
                                    for theta in (.11,.13)}))
        save(out/'summary.json',report)
    report['status']='COMPLETE_CORRELATION_RATE_AUDIT_NOT_CERTIFIED'; save(out/'summary.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--device',choices=['cuda'],required=True)
    p.parse_args(); torch.set_num_threads(1); run()
