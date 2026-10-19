"""Prepare independent small-system parents and branched futures; explicit release required."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
import yaml
from scripts.core.reduced_llg import ReducedLLG
from scripts.core.literature_config import quantity
from tests.literature_preflight_20260917.models import build
from tests.physical_audit_20260917.cold import strong_rotation
from tests.bauer_two_stage_20260919.common import allocation,dump
from .geometry import CoreGeometry
from .data import validate_manifest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def advance(model,s,alpha,theta,scales,cfg,duration,generators,dataset=None,trace=None):
    llg=ReducedLLG(model,alpha=alpha,theta=theta)
    dt=cfg['dt_fs']*1e-15/scales['time_s']
    count=round(duration*1000/cfg['dt_fs']);stride=round(cfg['save_ps']*1000/cfg['dt_fs'])
    if count<=0 or stride<1 or count%stride:raise ValueError('Invalid simulation time grid')
    if dataset is not None:dataset[:,0]=s.cpu().numpy().astype('f4')
    with torch.inference_mode():
        for step in range(1,count+1):
            dw=(torch.cat([llg.noise_increment(s[j:j+1],dt,g) for j,g in enumerate(generators)],0)
                if theta else torch.zeros_like(s))
            s,_=llg.step(s,dt,dw,method='heun')
            if step%stride==0:
                if not torch.isfinite(s).all() or float(abs(s.norm(dim=-1)-1).max())>1e-10:
                    raise RuntimeError('Nonfinite spins or norm loss')
                if dataset is not None:dataset[:,step//stride]=s.cpu().numpy().astype('f4')
                if trace is not None:
                    trace.append(dict(time_ps=step*cfg['dt_fs']/1000,
                        energy_per_spin=float(model.energy(s).item()/(s.numel()/3))))
                if step//stride%50==0:print('physical_ps',step*cfg['dt_fs']/1000,flush=True)
    return s


def run(phase,index,device,smoke=False):
    if not smoke:
        allocation(device)
        release=json.loads((HERE/'release.json').read_text())
        if not release.get('approved_for_generation'):raise RuntimeError('Physics release missing')
        for name,path in [('config',HERE/'config.yaml'),('layout',HERE/'layout.json'),
                          ('source',ROOT/'tests/literature_preflight_20260917/config.yaml')]:
            if release.get(name+'_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise RuntimeError('Release no longer matches '+name)
    elif device=='cuda':allocation(device)
    torch.set_num_threads(1)
    cfg=yaml.safe_load((HERE/'config.yaml').read_text())
    plan=json.loads((HERE/'layout.json').read_text())
    output=HERE/('smoke_library' if smoke else 'data');output.mkdir(exist_ok=True)
    if smoke:cfg.update(cold_duration_ps=.005,parent_preparation_ps=.005,future_duration_ps=.01,save_ps=.0025)
    row=None if phase=='cold' else plan['parents'][index]
    shape=tuple(cfg['cell_shapes'][index] if row is None else row['cell_shape'])
    label=f'{shape[0]}x{shape[1]}'
    if smoke:shape=(6,4)
    source=yaml.safe_load((ROOT/cfg['source_config']).read_text())
    source['systems']['miltat']['physics'].update(temperature_K=cfg['temperature_K'],alpha=cfg['alpha'])
    model,reduced,scales=build(source,'miltat',shape=shape)
    if phase=='cold':
        target=output/f'cold_{label}.pt'
        if target.exists():raise RuntimeError('Cold base already exists')
        s=model.uniform([1,1],device=device,dtype=torch.float64)
        preparation=source['systems']['miltat']['preparation']
        s=model.initial_texture(s,**{k:preparation[k] for k in ['radius_nm','width_nm']})
        s=strong_rotation(s,shape,cfg)
        trace=[];started=time.perf_counter()
        s=advance(model,s,cfg['alpha'],0.,scales,cfg,cfg['cold_duration_ps'],[],trace=trace)
        torch.save(dict(spins=s.cpu(),config=cfg,shape=shape,smoke=smoke),target)
        positions=(model.positions(s)*reduced['a_nm']).cpu().numpy()
        geometry=CoreGeometry(positions)
        energy=np.array([r['energy_per_spin'] for r in trace])
        dump(output/f'cold_{label}.json',dict(shape=shape,core=geometry.measure(s.cpu().numpy()),
            norm_error=float(abs(s.norm(dim=-1)-1).max()),
            torque_max=float(torch.linalg.cross(s,model.field(s)).norm(dim=-1).max()),
            energy_trace=trace,last_window_energy_span=float(np.ptp(energy[-min(101,len(energy)):])),
            seconds=time.perf_counter()-started,smoke=smoke,scientific_certification=False))
        return
    if not smoke:
        gate=json.loads((output/'cold_gate.json').read_text())
        if not gate['approved']:raise RuntimeError('Cold protocol gate failed')
        cold_path=output/f'cold_{label}.pt'
        if hashlib.sha256(cold_path.read_bytes()).hexdigest()!=gate['checkpoint_sha256'][label]:
            raise RuntimeError('Cold checkpoint changed after verification')
    s=torch.load(output/f'cold_{label}.pt',map_location=device,weights_only=False)['spins']
    parent_rng=torch.Generator(device=device).manual_seed(row['preparation_seed'])
    initial=advance(model,s,cfg['alpha'],reduced['theta'],scales,cfg,cfg['parent_preparation_ps'],[parent_rng])
    count=cfg['branches_per_parent'];s=initial.repeat(count,1,1,1,1)
    generators=[torch.Generator(device=device).manual_seed(seed) for seed in row['noise_seeds']]
    positions=(model.positions(initial)*reduced['a_nm']).cpu().numpy()
    geometry=CoreGeometry(positions)
    target=output/row['file'];partial=target.with_suffix('.partial.h5')
    if target.exists():raise RuntimeError('Parent already generated')
    physical=source['systems']['miltat']['physics']
    physics=dict(boundary='open',temperature_K=cfg['temperature_K'],
                 **{key+'_meV':quantity(physical[key]).m_as('meV') for key in ['J','K','D']},
                 moment_muB=quantity(physical['moment']).m_as('mu_B'),
                 a_angstrom=quantity(physical['a']).m_as('angstrom'),
                 gamma_per_second_tesla=quantity(source['gamma_convention']).m_as('1/(second*tesla)'),alpha=cfg['alpha'])
    with h5py.File(partial,'x') as h:
        h.attrs['schema']='miltat_parent_v1';h.attrs['complete']=False;h.attrs['smoke']=smoke
        for key in ['parent_id','split','preparation_group']:h.attrs[key]=row[key]
        h.attrs['physical_parameters_json']=json.dumps(physics);h.attrs['config_json']=json.dumps(cfg)
        h.attrs['reduced_json']=json.dumps(reduced);h.attrs['scales_json']=json.dumps(scales)
        h.attrs['preparation_seed']=row['preparation_seed'];h.attrs['scientific_certification']=False
        if not smoke:h.attrs['cold_checkpoint_sha256']=gate['checkpoint_sha256'][label]
        h.attrs['initial_core_json']=json.dumps(geometry.measure(initial.cpu().numpy()))
        h.create_dataset('initial',data=initial[0].cpu().numpy().astype('f4'))
        h.create_dataset('positions_nm',data=positions)
        h.create_dataset('noise_seeds',data=np.asarray(row['noise_seeds'],dtype='i8'))
        frames=round(cfg['future_duration_ps']/cfg['save_ps'])+1
        h.create_dataset('time_ps',data=np.arange(frames)*cfg['save_ps'])
        data=h.create_dataset('spins',(count,frames,*shape,2,3),dtype='f4',chunks=(1,1,*shape,2,3),compression='lzf')
        final=advance(model,s,cfg['alpha'],reduced['theta'],scales,cfg,cfg['future_duration_ps'],generators,data)
        h.attrs['final_core_json']=json.dumps([geometry.measure(x.cpu().numpy()) for x in final])
        h.attrs['complete']=True
    partial.replace(target)


def finalize(smoke=False):
    out=HERE/('smoke_library' if smoke else 'data')
    plan=json.loads((HERE/'layout.json').read_text())
    if smoke:plan['parents']=[r for r in plan['parents'] if (out/r['file']).exists()]
    # Do not publish a partial manifest: every expected parent must be complete.
    integrity=[]
    for row in plan['parents']:
        with h5py.File(out/row['file']) as h:
            if not h.attrs['complete']:raise RuntimeError('Incomplete parent')
            maximum=0.
            for start in range(0,h['spins'].shape[1],16):
                block=h['spins'][:,start:start+16]
                if not np.isfinite(block).all():raise RuntimeError('Nonfinite stored trajectory')
                maximum=max(maximum,float(abs(np.linalg.norm(block,axis=-1)-1).max()))
            if maximum>1e-6:raise RuntimeError('Invalid stored spin norm')
            integrity.append(dict(parent_id=row['parent_id'],max_stored_norm_error=maximum,
                                  branches=h['spins'].shape[0],frames=h['spins'].shape[1]))
            if smoke:
                row['design_cell_shape']=row['cell_shape'];row['cell_shape']=list(h['spins'].shape[2:4])
    dump(out/'manifest.json',dict(plan,smoke=smoke))
    validate_manifest(out,allow_smoke=smoke)
    dump(out/'complete.json',dict(parents=len(plan['parents']),smoke=smoke,integrity=integrity,scientific_certification=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['cold','parent','finalize'])
    p.add_argument('--index',type=int,default=0);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    finalize(a.smoke) if a.phase=='finalize' else run(a.phase,a.index,a.device,a.smoke)
