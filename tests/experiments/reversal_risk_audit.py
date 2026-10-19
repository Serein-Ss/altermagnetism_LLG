"""Frozen checkpoint reversal defect and supervised risk; no training or gate."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from scripts.model.gate_f import reference_path, path_interpolate
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.velocity_controls import CartesianPathNet
from tests.experiments.stable_flow_stage import free_frame_risk


def reversal_risks(direct, reversed_back, target):
    return dict(original=free_frame_risk(direct,target)[0],
                reversed=free_frame_risk(reversed_back,target)[0],
                symmetric=free_frame_risk((direct+reversed_back)*.5,target)[0],
                defect=free_frame_risk(direct,reversed_back)[0])


@torch.no_grad()
def run():
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available(): raise RuntimeError('Slurm CUDA required')
    cfg_path=Path('tests/conf/stable_flow_stage.json'); cfg=json.loads(cfg_path.read_text())
    if digest(cfg['data'])!=cfg['data_sha256']: raise ValueError('data changed')
    freeze=json.loads(Path(cfg['freeze']).read_text())
    out=Path('tests/results/reversal_risk_audit'); out.mkdir(exist_ok=False)
    report=dict(status='RUNNING',rows=[],config=cfg,config_sha256=digest(cfg_path),
        scope='Exploratory fixed-window supervised velocity audit. Not physical-law validation. Symmetric error <= mean(original,reversed) is an algebraic identity, NOT evidence of learned dynamics. Compare original and reversed separately.',
        code_sha256={p:digest(p) for p in [__file__,'scripts/model/gate_f.py','tests/experiments/velocity_controls.py','tests/experiments/stable_flow_stage.py']})
    with np.load(cfg['data']) as d: arrays={k:d[k] for k in ('train','condition','development','development_condition')}
    save(out/'summary.json',report)
    for seed in cfg['seeds']:
        cp=Path(f'tests/output/stable_window_learning/window11/{seed}/step_40000.pt')
        if digest(cp)!=freeze['checkpoint_sha256'][str(seed)]: raise ValueError('checkpoint changed')
        state=torch.load(cp,map_location='cuda',weights_only=False)
        if state['inference_weights']!='ema': raise ValueError('EMA required')
        net=CartesianPathNet().cuda().eval(); net.load_state_dict(state['model'])
        for si,(split,ck) in enumerate((('train','condition'),('development','development_condition'))):
            a=arrays[split]; n=len(a); w=cfg['windows_per_parent']; f=cfg['window_frames']
            starts=np.random.default_rng(cfg['random_seed']+si).integers(0,a.shape[1]-f+1,size=(n,w))
            parents=np.repeat(np.arange(n),w)
            windows=a[parents[:,None],starts.ravel()[:,None]+np.arange(f)]
            values={k:np.empty((n*w,len(cfg['tau']),f-1)) for k in ('original','reversed','symmetric','defect')}
            for off in range(0,n*w,cfg['batch_size']):
                stop=min(off+cfg['batch_size'],n*w)
                y=torch.as_tensor(windows[off:stop,:,None,:,None],device='cuda',dtype=torch.float32)
                c=torch.as_tensor(arrays[ck][parents[off:stop]],device='cuda',dtype=y.dtype)
                t=torch.linspace(0,1,a.shape[1],device='cuda')[:f][None].expand(len(y),-1)
                local=cfg['random_seed']+10000+si*100000+off
                z=reference_path(y[:,0],f,torch.Generator(device='cuda').manual_seed(local))
                for ti,tau in enumerate(cfg['tau']):
                    flow=y.new_full((len(y),),tau)
                    x,target,_=path_interpolate(z,y,flow,torch.Generator(device='cuda').manual_seed(local+1))
                    direct=net(x,flow,y[:,0],c,t,None,None)
                    rev=net(x.flip(-3),flow,y[:,0].flip(-3),c,t,None,None).flip(-3)
                    for k,v in reversal_risks(direct,rev,target).items(): values[k][off:stop,ti]=v.cpu().numpy()
            values={k:v.reshape(n,w,len(cfg['tau']),f-1) for k,v in values.items()}
            with np.load(f'tests/results/stable_flow_stage/{split}_seed{seed}.npz') as old:
                if not np.array_equal(starts,old['window_starts']): raise ValueError('different old windows')
                if not np.allclose(values['original'],old['squared_vector_error'],atol=1e-6,rtol=1e-6): raise ValueError('original replay changed')
            dest=out/f'{split}_seed{seed}.npz'
            np.savez_compressed(dest,**values,window_starts=starts,condition=arrays[ck],tau=cfg['tau'])
            report['rows'].append(dict(seed=seed,split=split,path=str(dest),sha256=digest(dest),checkpoint_sha256=digest(cp)))
            save(out/'summary.json',report)
    report['status']='COMPLETE_REVERSAL_RISK_NOT_CERTIFIED'; save(out/'summary.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--device',choices=['cuda'],required=True); p.parse_args()
    torch.set_num_threads(1); run()
