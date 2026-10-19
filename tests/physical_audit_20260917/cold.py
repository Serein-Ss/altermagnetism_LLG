"""Save full-domain zero-noise trajectories, separately from thermal pilots."""
import argparse
import json
import os
from pathlib import Path
import h5py
import numpy as np
import torch
import yaml
from scripts.core.reduced_llg import ReducedLLG
from tests.literature_preflight_20260917.models import build

BASE=Path(__file__).resolve().parent
PILOT=BASE.parent/'literature_preflight_20260917'


def strong_rotation(s, shape, settings):
    coordinates=torch.meshgrid(*[torch.arange(size,device=s.device,dtype=s.dtype) for size in shape],indexing='ij')
    modulation=torch.ones(shape,device=s.device,dtype=s.dtype)
    for coordinate,size in zip(coordinates,shape): modulation*=torch.cos(2*torch.pi*coordinate/size)
    angle=(settings['strong_rotation_center_degrees']+settings['strong_rotation_modulation_degrees']*modulation)*torch.pi/180
    angle=angle[None,...,None]
    result=s.clone(); x,z=s[...,0],s[...,2]
    result[...,0]=angle.cos()*x+angle.sin()*z
    result[...,2]=-angle.sin()*x+angle.cos()*z
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--system',required=True,choices=['bauer','miltat','hirst'])
    p.add_argument('--strong',action='store_true'); args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available(): raise RuntimeError('Scheduled CUDA required')
    torch.set_num_threads(1)
    cfg=yaml.safe_load((PILOT/'config.yaml').read_text()); settings=yaml.safe_load((BASE/'cold_settings.yaml').read_text())
    cell,r,scales=build(cfg,args.system); n=cfg['systems'][args.system]['numerics']
    with h5py.File(PILOT/'results'/args.system/'trajectory.h5') as f: initial=f['spins'][0,:1]
    s=torch.tensor(initial,device='cuda',dtype=torch.float64)
    unperturbed=s.clone()
    labels=['same_initial_as_thermal_pilot']
    if args.strong:
        s=strong_rotation(s,cell.shape,settings)
        labels=['spatially_modulated_y_rotation_20_to_50_degrees_strong_diagnostic']
    elif args.system in settings['tilted_systems']:
        a=np.deg2rad(settings['tilt_degrees']); rotation=s.new_tensor([[np.cos(a),0,-np.sin(a)],[0,1,0],[np.sin(a),0,np.cos(a)]])
        s=torch.cat((s,s@rotation),0); labels.append('rigid_y_rotation_10_degrees_diagnostic_not_paper_protocol')
    llg=ReducedLLG(cell,alpha=r['alpha'],theta=0.,equation_convention='bauer_ll' if args.system=='bauer' else 'gilbert')
    dt_fs=n['dt_fs']/settings['dt_divisor']; dt=dt_fs*1e-15/scales['time_s']
    steps=round(n['duration_ps']*1000/dt_fs); stride=round(n['save_ps']*1000/dt_fs)
    out=BASE/'results'/args.system/('cold_strong' if args.strong else 'cold'); out.mkdir(parents=True,exist_ok=False)
    start=s.clone(); norm=0.; max_rise=-float('inf'); previous=cell.energy(s)/s[0,...,0].numel()
    deviation=torch.acos((s*unperturbed).sum(-1).clamp(-1,1))*180/torch.pi
    energy_excess=(cell.energy(s)-cell.energy(unperturbed))/s[0,...,0].numel()
    energies=[]
    with h5py.File(out/'trajectory.h5','x') as f:
        frames=f.create_dataset('spins',(steps//stride+1,*s.shape),dtype='f8',compression='lzf')
        f.create_dataset('time_ps',data=np.arange(steps//stride+1)*n['save_ps'])
        frames[0]=s.cpu().numpy(); energies.append(previous.cpu().numpy())
        for k in range(1,steps+1):
            s,_=llg.step(s,dt,torch.zeros_like(s),method=settings['method'])
            if k%stride==0:
                e=cell.energy(s)/s[0,...,0].numel(); max_rise=max(max_rise,float((e-previous).max())); previous=e
                norm=max(norm,float((s.norm(dim=-1)-1).abs().max())); assert torch.isfinite(s).all()
                frames[k//stride]=s.cpu().numpy(); energies.append(e.cpu().numpy())
        f.create_dataset('energy_per_spin',data=np.array(energies)); f.attrs['trajectory_labels']=json.dumps(labels)
        f.attrs['temperature_K']=0.; f.attrs['noise']='exactly zero; no RNG used'; f.attrs['method']=settings['method']
    report=dict(system=args.system,trajectories=len(s),labels=labels,temperature_K=0.,noise='exactly_zero',
                duration_ps=n['duration_ps'],save_ps=n['save_ps'],dt_fs=dt_fs,method=settings['method'],
                max_unit_norm_error=norm,max_saved_energy_rise_per_spin=max_rise,
                final_rms_from_initial=(s-start).square().flatten(1).mean(1).sqrt().cpu().tolist(),
                initial_angle_mean_degrees=float(deviation.mean()),initial_angle_max_degrees=float(deviation.max()),
                initial_excess_energy_per_spin=energy_excess.cpu().tolist(),
                perturbation='20-50 degree smooth y rotation; identical within each basis cell' if args.strong else 'same initial and optional 10-degree control',
                job=os.environ['SLURM_JOB_ID'],production_enabled=False,
                note='T=0 still has damping. The tilted initial conditions are new diagnostic choices, not claimed paper protocols.')
    (out/'report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2),flush=True)
    assert norm<1e-10 and max_rise<1e-9


if __name__=='__main__': main()
