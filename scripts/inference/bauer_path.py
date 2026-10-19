"""Portable development checkpoint loading and native-grid generation."""
import argparse,json
from pathlib import Path
import h5py
import numpy as np
import torch
from scripts.model.bauer_path import ReversalGauge
from scripts.model.path_sampling import rollout,require_device


def load_checkpoint(path,device='cpu'):
    device=require_device(device)
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    net=ReversalGauge('reversal');net.load_state_dict(checkpoint['model']);net=net.to(device).eval()
    return net,checkpoint


def generate(net,initial,duration_ps,seed,flow_steps=128,generator=None):
    count=round(duration_ps/.1)
    if duration_ps<=0 or not np.isclose(count*.1,duration_ps,atol=1e-9,rtol=0):
        raise ValueError('duration must be a positive multiple of native0.1ps')
    if initial.ndim!=3 or initial.shape[-1]!=3 or not torch.isfinite(initial).all():
        raise ValueError('initial shape must be [branch,site,3]')
    if not torch.allclose(initial.norm(dim=-1),torch.ones_like(initial[...,0]),atol=2e-6,rtol=0):
        raise ValueError('initial spins must have unit norm')
    if flow_steps<1:raise ValueError('positive flow steps required')
    require_device(initial.device)
    generator=torch.Generator(device=initial.device).manual_seed(seed) if generator is None else generator
    paths=rollout(net,initial,generator,flow_steps,count)
    return paths,generator.get_state()


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--initial',required=True)
    p.add_argument('--output',required=True);p.add_argument('--duration-ps',type=float,default=1.)
    p.add_argument('--branches',type=int,default=16);p.add_argument('--seed',type=int,default=11)
    p.add_argument('--flow-steps',type=int,default=128);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    torch.set_num_threads(1);net,ck=load_checkpoint(a.checkpoint,a.device)
    initial=np.load(a.initial,allow_pickle=False)
    if initial.ndim!=2 or a.branches<1:raise ValueError('provide one [site,3] initial .npy and positive branch count')
    initial=torch.as_tensor(initial,device=a.device,dtype=torch.float32)[None].repeat(a.branches,1,1)
    paths,rng=generate(net,initial,a.duration_ps,a.seed,a.flow_steps)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(out,'x') as h:
        h.attrs.update(complete=False,checkpoint=str(a.checkpoint),step=ck['step'],model_contract=json.dumps(net.get_extra_state()),
            scientific_certification=False,physical_lag_ps=.1,requested_duration_ps=a.duration_ps,flow_steps=a.flow_steps,latent_seed=a.seed)
        h.create_dataset('spins',data=paths.cpu().numpy(),compression='lzf')
        h.create_dataset('time_ps',data=np.arange(paths.shape[1])*.1);h.create_dataset('rng_final',data=rng.cpu().numpy());h.attrs['complete']=True
    print(out)


if __name__=='__main__':main()
