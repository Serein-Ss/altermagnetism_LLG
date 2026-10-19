"""Opt-in literature-length runner with exact model/RNG state at chunk boundaries.

Not submitted by the bounded completion campaign. Requires explicit cost approval.
"""
import argparse
import json
import os
import time
import numpy as np
import torch
from scripts.workflow.bauer_campaign import digest, save
from tests.paper_work.completion.run import ROOT, load, model_load, require_gpu, initial
from tests.experiments.projected_midpoint_sampler import sample


@torch.no_grad()
def advance(model, ck, x, theta, rng, physical_steps, stride):
    temperature=x.new_full((len(x),),theta);lag=x.new_full((len(x),),ck['config']['lag'])
    axis=x.new_tensor([0.,0.,1.]).expand(len(x),-1);frames=[]
    assert physical_steps % stride == 0
    for step in range(physical_steps):
        x=sample(model,x,temperature,lag,axis,rng,ck['latent_scale'],32)
        if (step+1)%stride==0:frames.append(x.cpu().numpy())
    return x,np.stack(frames,axis=1)


def run(args):
    require_gpu();m=load();c=m['config'];checkpoint=c['frozen_checkpoint']
    model,ck=model_load(checkpoint,'cuda');state=m['states'][args.state]
    root=ROOT/'data'/f'literature_state{args.state}_seed{args.seed}_n{args.paths}'
    root.mkdir(exist_ok=True)
    manifest=dict(checkpoint=checkpoint,sha256=digest(checkpoint),state=state,seed=args.seed,
                  paths=args.paths,target_duration=750000.,chunk_duration=200.,save_dt=.1,
                  code_sha256=digest(__file__),campaign_sha256=digest(ROOT/'manifest.json'))
    mp=root/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    else:save(mp,manifest)
    rng=torch.Generator(device='cuda').manual_seed(args.seed)
    x=torch.tensor(initial(m,args.state),device='cuda',dtype=torch.float32).expand(args.paths,-1,-1).clone()
    x=torch.nn.functional.normalize(x,dim=-1)
    first=0
    # Only a complete .pt marker commits its sibling raw chunk. Ignore no gaps.
    for index in range(3750):
        marker=root/f'chunk{index:04d}.pt'
        if not marker.exists():first=index;break
        last=torch.load(marker,map_location='cpu',weights_only=False)
        assert last['manifest_sha256']==digest(mp)
        assert digest(root/f'chunk{index:04d}.npz')==last['data_sha256']
        x=last['spins'].cuda();rng.set_state(last['rng'])
        first=index+1
    for index in range(first,min(first+args.chunks,3750)):
        started=time.monotonic();start=x.cpu().numpy()
        x,frames=advance(model,ck,x,state['theta'],rng,20000,10)
        p=np.concatenate((start[:,None],frames),axis=1)
        if not np.isfinite(p).all() or np.max(abs(np.linalg.norm(p,axis=-1)-1))>=1e-5:
            raise ValueError('Invalid path')
        raw=root/f'chunk{index:04d}.npz';tmp=root/f'chunk{index:04d}.tmp.npz'
        np.savez_compressed(tmp,spins=p,reduced_time=index*200+np.arange(2001)*.1)
        os.replace(tmp,raw)
        marker=root/f'chunk{index:04d}.pt';temporary=marker.with_suffix('.tmp.pt')
        torch.save(dict(spins=x.cpu(),rng=rng.get_state().cpu(),data_sha256=digest(raw),
                        manifest_sha256=digest(mp),seconds=time.monotonic()-started),temporary)
        os.replace(temporary,marker)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--budget-approved',action='store_true',required=True)
    p.add_argument('--state',type=int,default=12);p.add_argument('--seed',type=int,default=3600000000)
    p.add_argument('--paths',type=int,default=64);p.add_argument('--chunks',type=int,default=1)
    p.add_argument('--device',choices=['cuda'],required=True);a=p.parse_args()
    if not 0<a.paths<=128 or not 0<a.chunks<=3750:raise ValueError('Invalid bounded run size')
    run(a)
