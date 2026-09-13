"""YAML-only reduced Nishino case A/B; raw data never self-certify."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import h5py
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT.parent))
from altermagnetism_LLG.scripts.core.literature_config import load_runtime, sha256
from altermagnetism_LLG.scripts.core.reduced_llg import ReducedLLG
from altermagnetism_LLG.scripts.literature.nishino_miyashita_2015.model import FreeMoments, case_alpha

PAPER='nishino_miyashita_2015'


def task_coordinates(config,phase,task):
    n=config.numerics; ext=n['validation_extension']
    repeats=ext['convergence_repetitions'] if phase=='convergence' else ext['repetitions']
    grid=n['numerical_choice']['theta_grid']
    if not 0<=task<2*len(grid)*repeats: raise ValueError('task index out of range')
    condition,rep=divmod(task,repeats); case_index,index=divmod(condition,len(grid))
    return ('A','B')[case_index],index,rep


def simulate(config,case,theta_index,rep,*,phase='primary',device='cpu',steps=None,moments=None):
    n=config.numerics; chosen=n['numerical_choice']; paper=n['paper_settings']; ext=n['validation_extension']
    theta=chosen['theta_grid'][theta_index]
    alpha=case_alpha(config.reduced,case,theta)
    model=FreeMoments(config.reduced)
    solver=ReducedLLG(model,alpha=alpha,theta=theta,equation_convention=config.reduced['equation_convention'])
    coarse_dt=paper['dt']; total=steps or paper['total_steps']; sites=moments or paper['moments']
    burn=paper['burn_steps'] if steps is None else total//2
    cadence=chosen['save_every'] if steps is None else max(1,total//8)
    methods=ext['methods'] if phase=='convergence' else ['paper_midpoint']
    factors=ext['factors'] if phase=='convergence' else [n.get('selected_factor',1)]
    seed_text=f"{chosen['seed']}:{phase}:{case}:{theta_index}:{rep}"
    seed=int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:7],'little')
    generator=torch.Generator(device=device).manual_seed(seed)
    state=torch.zeros((1,sites,3),dtype=torch.float64,device=device);state[...,2]=1
    states={(method,factor):state.clone() for method in methods for factor in factors}
    traces={key:[state[0,:,2].cpu().numpy().copy()] for key in states}
    vectors=[state[0].cpu().numpy().copy()] if phase!='convergence' else None
    errors={key:np.zeros(2) for key in states};norms={key:0. for key in states}
    sums={key:torch.zeros(sites,dtype=torch.float64,device=device) for key in states}
    counts={key:0 for key in states}; times=[0.]
    start=time.perf_counter()
    for step in range(total):
        # Same fine Wiener path drives every numerical variant. No shared draws
        # between particles or independent replicates.
        finest=max(factors)
        dw=torch.stack([solver.noise_increment(state,coarse_dt/finest,generator) for _ in range(finest)])
        for key in states:
            method,factor=key
            increments=dw.reshape(factor,finest//factor,*state.shape).sum(1)
            current=states[key]
            for subincrement in increments:
                current,raw_error=solver.step(current,coarse_dt/factor,subincrement,method=method,
                    tolerance=chosen['midpoint_tolerance'],max_iterations=chosen['midpoint_iterations'])
                errors[key]=np.maximum(errors[key],raw_error.cpu().numpy())
            states[key]=current
            norm=float((current.norm(dim=-1)-1).abs().max())
            norms[key]=max(norms[key],norm)
            if not torch.isfinite(current).all() or norm>ext['norm_tolerance']:
                raise RuntimeError('nonfinite state or norm gate failed')
            # Mean includes every coarse-time measurement after burn-in;
            # saved cadence does not alter mean estimation or integration.
            if step>=burn:
                sums[key]+=current[0,:,2];counts[key]+=1
            if (step+1)%cadence==0 or step+1==total:
                traces[key].append(current[0,:,2].cpu().numpy().copy())
        if (step+1)%cadence==0 or step+1==total:
            times.append((step+1)*coarse_dt)
            if vectors is not None: vectors.append(next(iter(states.values()))[0].cpu().numpy().copy())
        if (step+1)%10000==0:
            print(f'case={case} theta={theta} repeat={rep} step={step+1}/{total}',flush=True)
    elapsed=time.perf_counter()-start
    results={}
    for key,current in states.items():
        method,factor=key
        results[f'{method}/factor_{factor}']={'z':np.asarray(traces[key]),
            'per_moment_mean':(sums[key]/counts[key]).cpu().numpy(),
            'final_spins':current[0].cpu().numpy(),
            'raw_norm_errors':errors[key], 'norm_error':norms[key]}
    meta={'case':case,'theta_index':theta_index,'theta':theta,'repeat':rep,'seed':seed,
        'phase':phase,'alpha':alpha,'D':alpha*theta,'dt':coarse_dt,'total_steps':total,
        'burn_steps':burn,'moments':sites,'save_every':cadence,'device':device,'dtype':'float64',
        'rng':'torch.Generator independent components, SHA256 task seed',
        'elapsed_seconds':elapsed,'formal_settings':steps is None and moments is None,
        'equation_convention':config.reduced['equation_convention'],
        'unit_system':'reduced','numerics':n,'reduced':config.reduced}
    return meta,results,np.asarray(times),None if vectors is None else np.asarray(vectors)


def write_raw(destination,metadata,results,times,vectors=None):
    destination=Path(destination);temporary=destination.with_suffix('.h5.partial')
    if destination.exists() or temporary.exists(): raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(temporary,'x') as h5:
        h5.attrs['metadata']=json.dumps(metadata)
        h5.create_dataset('tau',data=times)
        for key,row in results.items():
            group=h5.create_group(key)
            for name in ['z','per_moment_mean','final_spins','raw_norm_errors']:
                group.create_dataset(name,data=row[name],compression='gzip',shuffle=True)
            group.attrs['norm_error']=row['norm_error']
        if vectors is not None: h5.create_dataset('spins',data=vectors,compression='gzip',shuffle=True)
        h5.attrs['complete']=True
    temporary.rename(destination)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--project-root',type=Path,default=ROOT)
    p.add_argument('--run-id',required=True)
    p.add_argument('--phase',choices=['convergence','primary','smoke','timing'],required=True)
    p.add_argument('--task',type=int,default=0)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args(); config=load_runtime(args.config)
    case,index,rep=task_coordinates(config,'convergence' if args.phase in ['convergence','timing'] else 'primary',args.task)
    steps=32 if args.phase=='smoke' else config.numerics['validation_extension']['timing_steps'] if args.phase=='timing' else None
    phase='convergence' if args.phase=='timing' else args.phase
    meta,results,times,vectors=simulate(config,case,index,rep,phase=phase,device=args.device,
        steps=steps,moments=8 if args.phase=='smoke' else None)
    meta.update({'config_sha256':sha256(args.config),'run_id':args.run_id,
                 'code_commit':config.numerics.get('code_commit','unfrozen'),
                 'code_sha256':config.numerics.get('code_sha256','unfrozen')})
    output=args.project_root/'data/literature_reproduction'/PAPER/'raw'/(args.run_id+'.partial')/f'task_{args.task:04d}.h5'
    write_raw(output,meta,results,times,vectors)
    print(json.dumps({'output':str(output),'elapsed_seconds':meta['elapsed_seconds'],
                     'formal_settings':meta['formal_settings'],'data_sha256':sha256(output)}),flush=True)


if __name__=='__main__': main()
