"""Identical central perturbation, different open boxes; keep every outcome."""
import argparse
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch
import yaml
from scripts.core.reduced_llg import ReducedLLG
from tests.literature_preflight_20260917.models import build
from tests.physical_audit_20260917.cold import strong_rotation
from tests.miltat_dataset_20260920.topology import triangles, charge
from tests.bauer_two_stage_20260919.common import allocation, dump
from .boundary import centered_crop, HERE, ROOT


def run(index, device, smoke=False):
    if not smoke or device=='cuda': allocation(device)
    torch.set_num_threads(1)
    cfg = yaml.safe_load((HERE/'relaxation.yaml').read_text())
    source = yaml.safe_load((ROOT/'tests/literature_preflight_20260917/config.yaml').read_text())
    source['systems']['miltat']['physics'].update(temperature_K=0.,alpha=cfg['alpha'])
    shape = tuple(cfg['cell_shapes'][index])
    if smoke: shape=(6,4); cfg.update(duration_ps=.01,save_ps=.0025)
    model, reduced, scales = build(source,'miltat',shape=shape)
    if smoke:
        base = model.uniform([1,1],device=device)
        initial = strong_rotation(base,shape,cfg)
    else:
        parent = torch.load(ROOT/cfg['source_parent'],map_location='cpu',weights_only=False)['spins'].to(device)
        full_shape = tuple(parent.shape[1:3]); crop = centered_crop(full_shape,shape)
        idx = (slice(None),*crop,slice(None),slice(None))
        base = parent[idx].clone()
        initial = strong_rotation(parent,full_shape,cfg)[idx].clone()
    s = initial.clone(); pos = (model.positions(s)*reduced['a_nm']).cpu().numpy(); tri=triangles(pos)
    llg = ReducedLLG(model,alpha=cfg['alpha'],theta=0.)
    dt = cfg['dt_fs']*1e-15/scales['time_s']
    steps=round(cfg['duration_ps']*1000/cfg['dt_fs']); stride=round(cfg['save_ps']*1000/cfg['dt_fs'])
    assert steps%stride==0
    out=HERE/('smoke' if smoke else 'relaxation')/f'size_{index}';out.mkdir(parents=True,exist_ok=False)
    angle = torch.acos((base*initial).sum(-1).clamp(-1,1))*180/torch.pi
    rows=[];started=time.perf_counter()
    with torch.inference_mode(), h5py.File(out/'trajectory.h5','x') as h:
        h.attrs['complete']=False;h.attrs['config_json']=json.dumps(cfg);h.attrs['scales_json']=json.dumps(scales)
        h.attrs['reduced_json']=json.dumps(reduced);h.attrs['temperature_K']=0.;h.attrs['noise']='exactly_zero'
        h.create_dataset('positions_nm',data=pos)
        h.create_dataset('time_ps',data=np.arange(steps//stride+1)*cfg['save_ps'])
        frames=h.create_dataset('spins',(steps//stride+1,*s.shape[1:]),dtype='f4',chunks=(1,*s.shape[1:]),compression='lzf')
        def save(j):
            x=s[0].cpu().numpy();diag=charge(x,tri)
            if diag['stored_norm_error']>1e-10:raise RuntimeError('Spin norm failure')
            frames[j]=x
            core=x[...,2]>=.5; count=int(core.sum())
            torque=torch.linalg.cross(s,model.field(s))
            rows.append(dict(time_ps=j*cfg['save_ps'],**diag,core_sites=count,
                core_area_radius_nm=float(np.sqrt(count*np.sqrt(3)/2*reduced['a_nm']**2/np.pi)),
                energy_per_spin=float(model.energy(s).item()/(s.numel()/3)),
                torque_rms=float(torque.square().mean().sqrt()),
                torque_max=float(torque.norm(dim=-1).max()),
                warning='core area includes all sz>=0.5 sites, inspect texture for disconnected components'))
            h.attrs['frames_written']=j+1;h.flush()
        save(0)
        for k in range(1,steps+1):
            s,_=llg.step(s,dt,torch.zeros_like(s),method='heun')
            if k%stride==0:
                save(k//stride)
                if k//stride%10==0:print(index,k//stride,time.perf_counter()-started,flush=True)
        h.attrs['complete']=True
    energy=np.array([r['energy_per_spin'] for r in rows]); tail=energy[-min(21,len(energy)):]
    dump(out/'metrics.json',dict(config=cfg,shape=list(shape),rows=rows,smoke=smoke,
        perturbation_mean_degrees=float(angle.mean()),perturbation_max_degrees=float(angle.max()),
        max_saved_energy_rise=float(np.diff(energy).max()),last_window_energy_span_per_spin=float(np.ptp(tail)),
        seconds=time.perf_counter()-started,scientific_certification=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,choices=range(5),default=0)
    p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.index,a.device,a.smoke)
