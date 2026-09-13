"""Dedicated reduced, anisotropic, zero-drive entry. Never the old V3 solver.

This entry currently permits software validation and equilibrium diagnostics.
Production and path pilots require independently certified equilibrium pools;
uniform/random calibration chains must not be mislabeled as equilibrium paths.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import h5py
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.core.literature_config import load_runtime,sha256
from scripts.core.reduced_llg import ReducedLLG
from scripts.core.streaming_llg import simulate_resumable
from scripts.literature.gomonay_2024.model import DoubleLayer

DRIVES=('zeeman','field_gradient','sot','stt','current','pulse','tip')


def validate_config(c):
    if set(c['drives'])!=set(DRIVES) or any(np.any(np.asarray(c['drives'][k])!=0) for k in DRIVES):
        raise ValueError('all seven external drive declarations must be explicitly zero')
    if c['role'] not in ('software_validation','equilibrium_diagnostic'):
        raise RuntimeError('production/path pilot blocked: certified preparation pool and P1-P5 not available')
    if c.get('production_enabled') is not False: raise ValueError('production must remain disabled')
    if c['orientation'] not in ('100','110') or len(c['shape'])!=2 or min(c['shape'])<2:
        raise ValueError('invalid cell geometry')
    if len(c['initializations'])!=c['batch'] or not set(c['initializations'])<=set(('positive','negative','random')):
        raise ValueError('one declared calibration initialization per chain')
    if len(c['noise_seeds'])!=c['batch'] or len(set(c['noise_seeds']))!=c['batch']:
        raise ValueError('independent trajectory seeds required')
    if c['alpha']<=0 or c['theta']<0: raise ValueError('invalid thermal parameters')


def run(config,path,*,device='cpu',stop_after=None):
    config=Path(config);c=yaml.safe_load(config.read_text());validate_config(c)
    if device=='cuda' and (not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available()):
        raise RuntimeError('CUDA requires Slurm GPU allocation')
    material=Path(c['material_config'])
    if not material.is_absolute(): material=ROOT/material
    r=load_runtime(material).reduced
    if r['K_DW']<=0: raise ValueError('zero-field ordered model must use positive K_DW')
    model=DoubleLayer(r,c['shape'],wall=True,periodic=tuple(c['periodic']),orientation=c['orientation'])
    llg=ReducedLLG(model,alpha=c['alpha'],theta=c['theta'])
    s=model.ground_state(batch=c['batch'],device=device)
    for i,mode in enumerate(c['initializations']):
        if mode=='negative': s[i]*=-1
        elif mode=='random':
            g=torch.Generator(device=device).manual_seed(c['initial_seed']+i)
            state=torch.randn(s[i].shape,dtype=s.dtype,device=device,generator=g)
            s[i]=state/state.norm(dim=-1,keepdim=True)
    code={str(f.relative_to(ROOT)):sha256(f) for f in
          [Path(__file__),ROOT/'scripts/core/reduced_llg.py',ROOT/'scripts/core/cell_hamiltonian.py',
           ROOT/'scripts/core/streaming_llg.py',ROOT/'scripts/literature/gomonay_2024/model.py']}
    identity=dict(config=sha256(config),material=sha256(material),code=code)
    signs=s.new_tensor([1.,-1.]*(model.basis//2))
    def observe(state):
        n=(state*signs[None,None,None,:,None]).mean(3)
        mean=n.mean((1,2))
        spectrum=torch.fft.fft2(n,dim=(1,2)).abs().square().sum(-1)
        cells=n.shape[1]*n.shape[2]
        corr=torch.fft.ifft2(spectrum,dim=(1,2)).real/cells
        return dict(neel=mean,neel_spatial_std=n.std((1,2),unbiased=False),
                    neel_local_norm=n.norm(dim=-1).mean((1,2)),
                    cell_neel_structure_factor=spectrum/cells,
                    cell_neel_correlation_x=corr[:,:,0],cell_neel_correlation_y=corr[:,0,:])
    state,done=simulate_resumable(path,s,llg,c,identity=identity,observer=observe,stop_after=stop_after)
    if done:
        with h5py.File(path,'r+') as h:
            if not np.isfinite(h['max_norm_error'][:]).all() or h['max_norm_error'][:].max()>1e-10:
                raise FloatingPointError('long-trajectory unit-sphere error exceeds 1e-10; raw retained, no certificate')
            h.attrs.update(schema='gomonay_zero_field_v1_diagnostic',role=c['role'],
                unit_system='reduced',physics_certified=False,production_enabled=False,
                anisotropy_key='K_DW',material_json=json.dumps(r),drives_json=json.dumps(c['drives']),
                geometry_json=json.dumps(dict(shape=c['shape'],orientation=c['orientation'],
                    periodic=c['periodic'],basis=model.basis,
                    cell_vectors=[[1,0],[0,1]] if model.basis==2 else [[1,1],[-1,1]])),
                split='calibration_only_not_training',events_status='not_classified_no_basin_certificate')
            for name,data in dict(coordinates=model.coordinates().numpy(),
                 sublattice_sign=signs.cpu().numpy(),sample_weight=np.ones(len(s)),
                 parent_chain_id=np.array([f'{c["run_id"]}_chain_{i}'.encode() for i in range(len(s))]),
                 initial_state_id=np.array([f'{c["run_id"]}_initial_{i}'.encode() for i in range(len(s))])).items():
                if name not in h: h.create_dataset(name,data=data)
        manifest=dict(file=str(Path(path).resolve()),sha256=sha256(path),identity=identity,
                      status='complete_not_physics_certified',config=c)
        Path(str(path)+'.manifest.json').write_text(json.dumps(manifest,indent=2))
    return state,done


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    a=p.parse_args();_,done=run(a.config,a.output,device=a.device)
    if not done: raise SystemExit(85)


if __name__=='__main__': main()
