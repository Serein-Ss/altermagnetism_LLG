"""Predeclared six-standard-error RNG diagnostics through the actual interface."""
import argparse
import json
from pathlib import Path
import sys
import math
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.core.reduced_llg import ReducedLLG,BondHamiltonian
from scripts.core.literature_config import sha256


def audit(device='cpu'):
    import os
    if device=='cuda' and (not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available()):
        raise RuntimeError('allocated GPU required')
    llg=ReducedLLG(BondHamiltonian([],[]),alpha=.1,theta=.3)
    s=torch.zeros((32,10417,3),dtype=torch.float64,device=device)
    g=torch.Generator(device=device).manual_seed(2026091101)
    x=llg.noise_increment(s,.005,g)/math.sqrt(.005)
    y=llg.noise_increment(s,.005,g)/math.sqrt(.005)
    replay=llg.noise_increment(s,.005,torch.Generator(device=device).manual_seed(2026091101))/math.sqrt(.005)
    scaled=llg.noise_increment(s,.02,torch.Generator(device=device).manual_seed(2026091101))/math.sqrt(.02)
    a=x.cpu().numpy();b=y.cpu().numpy();n=a.size
    rows={}
    def metric(name,value,tolerance):
        rows[name]=dict(value=float(value),lower=-tolerance,upper=tolerance,pass_gate=bool(abs(value)<=tolerance))
    metric('mean',a.mean(),6/math.sqrt(n))
    metric('variance_minus_one',a.var(ddof=1)-1,6*math.sqrt(2/(n-1)))
    for i,j in ((0,1),(0,2),(1,2)):
        metric(f'component_{i}_{j}',np.corrcoef(a[...,i].ravel(),a[...,j].ravel())[0,1],6/math.sqrt(n/3))
    metric('adjacent_sites',np.corrcoef(a[:,:-1].ravel(),a[:,1:].ravel())[0,1],6/math.sqrt(a[:,:-1].size))
    metric('successive_steps',np.corrcoef(a.ravel(),b.ravel())[0,1],6/math.sqrt(n))
    metric('seed_replay_max_error',float((x-replay).abs().max()),0.)
    metric('dt_scaling_max_error',float((x-scaled).abs().max()),1e-14)
    s=torch.tensor([[[0.,0.,1.]]],dtype=torch.float64,device=device)
    zero=ReducedLLG(BondHamiltonian([],[],field=(.2,0.,0.)),alpha=.1,theta=0.)
    left,_=zero.step(s,.005,torch.zeros_like(s));right,_=zero.step(s,.005,torch.ones_like(s))
    metric('theta_zero_noise_effect',float((left-right).abs().max()),0.)
    return dict(scope='gilbert_noise_interface_only_not_interacting_equilibrium',components=n,
        seed=2026091101,device=device,torch=torch.__version__,numpy=np.__version__,
        threshold_design='six_standard_error_screen_frozen_before_draw_not_proof_of_independence',
        code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'scripts/core/reduced_llg.py']},
        metrics=rows,status='pass' if all(r['pass_gate'] for r in rows.values()) else 'fail',production_enabled=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--device',default='cpu',choices=('cpu','cuda'));a=p.parse_args()
    result=audit(a.device);a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
    if result['status']!='pass': raise SystemExit(2)
