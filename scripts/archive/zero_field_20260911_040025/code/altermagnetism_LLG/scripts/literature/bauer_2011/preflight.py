"""R2 prerequisites only. Short coupled paths do not certify reversal statistics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import h5py
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT.parent))
from altermagnetism_LLG.scripts.core.literature_config import load_runtime,sha256
from altermagnetism_LLG.scripts.core.reduced_llg import ReducedLLG
from altermagnetism_LLG.scripts.literature.bauer_2011.model import OpenChain


def static_checks(config):
    model=OpenChain(config.reduced)
    generator=torch.Generator().manual_seed(20260910)
    s=torch.randn((2,7,3),generator=generator,dtype=torch.float64,requires_grad=True)
    derivative=-torch.autograd.grad(model.energy(s).sum(),s)[0]
    torch.testing.assert_close(model.field(s),derivative,atol=1e-10,rtol=1e-10)
    up=torch.zeros(1,7,3,dtype=torch.float64);up[...,2]=1
    expected=-6*model.exchange-7*model.anisotropy
    assert abs(model.energy(up).item()-expected)<1e-12
    field=model.field(up)
    assert abs(field[0,0,2].item()-(model.exchange+2*model.anisotropy))<1e-12
    assert abs(field[0,3,2].item()-(2*model.exchange+2*model.anisotropy))<1e-12
    torque=torch.linalg.cross(up,field).abs().max().item()
    assert torque==0
    return dict(status='pass',field_autograd_max_abs=float((model.field(s)-derivative).abs().max()),
        open_bonds=6,ground_energy=expected,ground_torque=torque,
        scope='open-chain Hamiltonian and collinear static state only; no dynamical certification')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if not a.smoke and not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('timing requires Slurm')
    config=load_runtime(a.config);checks=static_checks(config)
    settings=config.numerics['preflight_choice'];steps=16 if a.smoke else settings['coarse_steps']
    length=config.numerics['paper_settings']['length'];batch=settings['independent_chains']
    dts=config.numerics['numerical_choice']['dt_scan'];dt=max(dts)
    factors=[round(dt/x) for x in dts]
    assert factors==[1,2,4]
    model=OpenChain(config.reduced)
    solver=ReducedLLG(model,alpha=config.reduced['alpha'],theta=config.reduced['theta'],equation_convention='bauer_ll')
    state=torch.zeros(batch,length,3,dtype=torch.float64);state[...,2]=1
    states={factor:state.clone() for factor in factors}
    frames={factor:[state.numpy().copy()] for factor in factors}
    errors={factor:np.zeros(2) for factor in factors};times=[0.]
    generator=torch.Generator().manual_seed(config.numerics['numerical_choice']['seed'])
    start=time.perf_counter()
    for step in range(steps):
        fine=torch.stack([solver.noise_increment(state,dt/4,generator) for _ in range(4)])
        for factor in factors:
            for dw in fine.reshape(factor,4//factor,*state.shape).sum(1):
                states[factor],error=solver.step(states[factor],dt/factor,dw,method=settings['method'])
                errors[factor]=np.maximum(errors[factor],error.numpy())
            if not torch.isfinite(states[factor]).all() or (states[factor].norm(dim=-1)-1).abs().max()>1e-10:
                raise RuntimeError('nonfinite/norm failure')
        if (step+1)%settings['save_every']==0 or step+1==steps:
            times.append((step+1)*dt)
            for factor in factors: frames[factor].append(states[factor].numpy().copy())
    elapsed=time.perf_counter()-start
    code_files=['scripts/core/reduced_llg.py','scripts/core/literature_config.py',
        'scripts/literature/bauer_2011/model.py','scripts/literature/bauer_2011/preflight.py']
    hashes={file:sha256(ROOT/file) for file in code_files}
    meta=dict(run_id=a.run_id,stage='R2_preflight',strict_reproduction=False,production_enabled=False,
        config_sha256=sha256(a.config),code_sha256=hashes,code_commit=os.environ.get('LLG_CODE_COMMIT','unfrozen'),device='cpu',dtype='float64',
        seed=config.numerics['numerical_choice']['seed'],coarse_steps=steps,coarse_dt=dt,
        reduced=config.reduced,numerics=config.numerics,elapsed_seconds=elapsed,
        integration_note='Heun is a project diagnostic choice, not the paper weak Runge-Kutta algorithm.',
        remaining=['audit/implement paper weak RK or independently justify alternative',
            'long-time step convergence','source length/temperature points','at least 500 completed reversal events per condition'])
    base=a.project_root/'data/literature_reproduction/bauer_2011/raw'
    partial=base/(a.run_id+'.partial');final=base/a.run_id
    if final.exists(): raise FileExistsError(final)
    partial.mkdir(parents=True)
    path=partial/'coupled_preflight.h5'
    with h5py.File(path,'x') as h5:
        h5.attrs['metadata']=json.dumps(meta);h5.create_dataset('tau',data=times)
        for factor in factors:
            group=h5.create_group(f'factor_{factor}')
            group.create_dataset('spins',data=np.asarray(frames[factor]),compression='gzip',shuffle=True)
            group.create_dataset('raw_norm_max_errors',data=errors[factor])
        h5.attrs['complete']=True
    with h5py.File(path) as h5: assert h5.attrs['complete'] and len(h5['tau'])==len(times)
    partial.rename(final)
    output=a.project_root/'output/literature_reproduction/bauer_2011'/a.run_id
    output.mkdir(parents=True,exist_ok=True)
    report=dict(**meta,static_checks=checks,software_status='pass',
        physical_certification='not_evaluated',raw_sha256=sha256(final/path.name),
        full_duration_estimate_seconds=elapsed*config.numerics['paper_settings']['duration']/(steps*dt),
        estimate_note='Same chain batch and three coupled dt variants only; excludes lifetime-condition and event-count expansion')
    with (output/'report.json').open('x') as stream: json.dump(report,stream,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
