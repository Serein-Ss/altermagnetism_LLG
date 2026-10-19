"""Add 0.3125fs paths by bridging the frozen 0.625fs Wiener increments."""
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
from tests.bauer_two_stage_20260919.common import allocation, dump
from tests.miltat_dataset_20260920.topology import triangles, charge

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=HERE.parent/'miltat_dataset_20260920'


def bridge(dw, eta, interval):
    delta=eta*(interval/4)**.5
    return dw*.5+delta,dw*.5-delta


def run(index,device,smoke=False):
    if not smoke or device=='cuda':allocation(device)
    torch.set_num_threads(1)
    cfg=yaml.safe_load((OLD/'convergence.yaml').read_text())
    source=yaml.safe_load((ROOT/cfg['source_config']).read_text())
    source['systems']['miltat']['physics'].update(temperature_K=cfg['temperature_K'],alpha=cfg['alpha'])
    if smoke:cfg.update(duration_ps=.01,save_ps=.0025)
    model,reduced,scales=build(source,'miltat',shape=(6,4) if smoke else None)
    if smoke:s=model.uniform([1,1],device=device)
    else:s=torch.load(ROOT/cfg['initial_checkpoint'],map_location='cpu',weights_only=False)['spins'].to(device)
    old_dt=cfg['coarse_dt_fs']/4*1e-15/scales['time_s']
    steps=round(cfg['duration_ps']*1000/cfg['coarse_dt_fs']);stride=round(cfg['save_ps']*1000/cfg['coarse_dt_fs'])
    rng=torch.Generator(device=device).manual_seed(cfg['branch_seeds'][index])
    bridge_seed=260920900+index
    extra=torch.Generator(device=device).manual_seed(bridge_seed)
    llg=ReducedLLG(model,alpha=reduced['alpha'],theta=reduced['theta'])
    pos=(model.positions(s)*reduced['a_nm']).cpu().numpy();tri=triangles(pos)
    out=HERE/('smoke' if smoke else 'data')/f'branch_{index}';out.mkdir(parents=True,exist_ok=False)
    oldroot=OLD/('smoke' if smoke else 'convergence')/f'branch_{index}'
    previous=json.loads((oldroot/'metrics.json').read_text())['rows']
    replay=s.clone();rows=[];started=time.perf_counter()
    with torch.inference_mode(), h5py.File(oldroot/'paths.h5') as old,h5py.File(out/'paths.h5','x') as h:
        assert old.attrs['complete']
        assert int(old.attrs['noise_seed'])==cfg['branch_seeds'][index]
        h.attrs['complete']=False;h.attrs['dt_fs']=cfg['coarse_dt_fs']/8
        h.attrs['bridge_seed']=bridge_seed;h.attrs['parent_noise_seed']=cfg['branch_seeds'][index]
        h.attrs['source_config_json']=json.dumps(cfg);h.attrs['scales_json']=json.dumps(scales)
        h.attrs['noise_coupling']='Each frozen 0.625fs increment split with independent Brownian bridge'
        h.create_dataset('positions_nm',data=pos);h.create_dataset('time_ps',data=old['time_ps'][:])
        d=h.create_dataset('spins',(len(old['time_ps']),*s.shape[1:]),dtype='f4',chunks=(1,*s.shape[1:]),compression='lzf')
        def save(j):
            array=s[0].cpu().numpy();diag=charge(array,tri)
            if diag['stored_norm_error']>1e-10:raise RuntimeError('Spin norm failed')
            d[j]=array;energy=float(model.energy(s).item()/(s.numel()/3))
            row=dict(time_ps=j*cfg['save_ps'],**diag,energy_per_spin=energy,
                mean_sz=float(array[...,2].mean()),core_sites=int((array[...,2]>=.5).sum()),
                spin_rmse_vs0625=float(np.sqrt(np.mean((array-old['spins'][2,j])**2))),
                energy_difference_vs0625=energy-previous[j]['levels'][2]['energy_per_spin'])
            rows.append(row);h.attrs['frames_written']=j+1;h.flush()
        save(0)
        for k in range(1,steps+1):
            # Preserve original generator, tensor shape, dtype, and four-draw call order.
            increments=[llg.noise_increment(s,old_dt,rng) for _ in range(4)]
            for dw in increments:
                eta=torch.randn(s.shape,device=device,dtype=s.dtype,generator=extra)
                for half in bridge(dw,eta,old_dt):s,_=llg.step(s,old_dt/2,half,method='heun')
                if k<=stride:replay,_=llg.step(replay,old_dt,dw,method='heun')
            if k==stride:
                error=float(np.max(np.abs(replay[0].cpu().numpy()-old['spins'][2,1])))
                h.attrs['replay_max_error_first_saved_frame']=error
                if error>2e-6:raise RuntimeError('Frozen Wiener path replay failed; cannot claim paired refinement')
            if k%stride==0:
                save(k//stride)
                if k//stride%10==0:print(index,k//stride,time.perf_counter()-started,flush=True)
        h.attrs['complete']=True
    dump(out/'metrics.json',dict(rows=rows,dt_fs=cfg['coarse_dt_fs']/8,smoke=smoke,
        replay_max_error_first_saved_frame=error,seconds=time.perf_counter()-started,
        scientific_certification=False,source=str(oldroot.relative_to(ROOT))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,choices=range(4),default=0)
    p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.index,a.device,a.smoke)
