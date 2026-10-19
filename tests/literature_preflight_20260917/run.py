"""Three bounded, coupled-dt numerical pilots. Never certifies a training dataset."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import h5py
import numpy as np
import torch
import yaml
from scripts.core.reduced_llg import ReducedLLG
from tests.literature_preflight_20260917.models import build

BASE=Path(__file__).resolve().parent


def verify_sources():
    manifest=json.loads((BASE/'source_hashes.json').read_text())
    for path,expected in manifest.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Source changed since submission: '+path)
    return manifest


def observables(model,s):
    axes=tuple(range(1,s.ndim-1))
    values={'magnetization':s.mean(axes).cpu().numpy(),
            'energy_per_spin':(model.energy(s)/s[0,...,0].numel()).cpu().numpy()}
    if hasattr(model,'order_parameters'):
        values['neel']=model.order_parameters(s)['neel'].cpu().numpy()
    if hasattr(model,'positions'):
        pos=model.positions(s)*model.reduced['a_nm']
        w=(s[...,2]>=.5).to(s.dtype)
        denom=w.flatten(1).sum(1)
        values['core_sites']=denom.cpu().numpy()
        values['core_center_nm']=((w[...,None]*pos).flatten(1,-2).sum(1)/denom[:,None].clamp_min(1)).cpu().numpy()
        values['core_center_nm'][denom.cpu().numpy()==0]=np.nan
    return values


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--system',choices=['bauer','miltat','hirst'],required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],required=True)
    parser.add_argument('--smoke',action='store_true',help='Tiny CPU I/O test, not a literature trajectory')
    args=parser.parse_args()
    if args.device=='cuda' and (not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available()):
        raise RuntimeError('CUDA pilot requires a real Slurm GPU allocation')
    config=yaml.safe_load((BASE/'config.yaml').read_text())
    manifest=verify_sources()
    if args.device=='cpu' and not args.smoke:
        raise RuntimeError('Full-size pilots must use the scheduled GPU runner')
    config=copy.deepcopy(config)
    row=config['systems'][args.system]
    if args.smoke:
        row['geometry']['shape']={'bauer':[6],'miltat':[4,3],'hirst':[3,3,3]}[args.system]
        row['numerics'].update(batch=2,duration_ps=row['numerics']['dt_fs']*.004,
                               save_ps=row['numerics']['dt_fs']*.002)
        if args.system=='miltat': row['preparation']['relax_ps']=0
    out=BASE/'results'/(args.system+'_cpu_smoke' if args.smoke else args.system)
    out.mkdir(parents=True,exist_ok=False)
    model,r,scales=build(config,args.system)
    n=row['numerics']; dtype=torch.float64
    if args.system=='hirst':
        s=model.ground_state(batch=n['batch'],device=args.device,dtype=dtype)
    else:
        s=model.uniform([1]*model.basis,batch=n['batch'],device=args.device,dtype=dtype)
        if args.system=='miltat':
            s=model.initial_texture(s,**{k:row['preparation'][k] for k in ['radius_nm','width_nm']})
    dt=n['dt_fs']*1e-15/scales['time_s']
    steps=round(n['duration_ps']*1000/n['dt_fs'])
    stride=round(n['save_ps']*1000/n['dt_fs'])
    assert steps%stride==0 and stride>0
    llg=ReducedLLG(model,alpha=r['alpha'],theta=r['theta'],
                   equation_convention='bauer_ll' if args.system=='bauer' else 'gilbert')
    initial_energy=model.energy(s).cpu().tolist()
    if args.system=='miltat':
        relax=ReducedLLG(model,alpha=r['alpha'],theta=0.)
        for _ in range(round(row['preparation']['relax_ps']*1000/n['dt_fs'])):
            s,_=relax.step(s,dt,torch.zeros_like(s),method='heun')
    states=[s.clone() for _ in (1,2,4)]
    max_raw=[0.,0.,0.]; max_norm=[0.,0.,0.]
    seed=2026091700+['bauer','miltat','hirst'].index(args.system)
    generator=torch.Generator(device=args.device).manual_seed(seed)
    metadata={'scope':config['scope'],'system':args.system,'config':row,'scales':scales,
              'reduced':r,'gamma_convention':config['gamma_convention'] if args.system!='bauer' else None,
              'source_hashes':manifest,'torch':torch.__version__,'numpy':np.__version__,
              'device':torch.cuda.get_device_name() if args.device=='cuda' else 'cpu',
              'seed':seed,'cpu_io_smoke':args.smoke,
              'initial_energy_reduced_before_preparation':initial_energy,
              'production_enabled':False,'scientific_status':'NOT_CERTIFIED',
              'dt_comparison':'same four fine Wiener increments summed for coarse levels; not independent reference ensembles'}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    start=time.perf_counter()
    with h5py.File(out/'trajectory.h5','x') as h:
        h.attrs['metadata']=json.dumps(metadata)
        h.attrs['time_unit']='ps'; h.attrs['energy_unit']='reduced_E0_in_metadata'
        data=h.create_dataset('spins',shape=(steps//stride+1,*s.shape),dtype='f8',
                              chunks=(1,1,*s.shape[1:]),compression='lzf')
        data.attrs['dt_level']=4
        h.create_dataset('time_ps',data=np.arange(steps//stride+1)*n['save_ps'])
        all_obs=[[],[],[]]
        def save(index):
            data[index]=states[-1].cpu().numpy()
            for k,state in enumerate(states): all_obs[k].append(observables(model,state))
            h.flush()
        save(0)
        for step in range(1,steps+1):
            dw=torch.randn((4,*s.shape),device=s.device,dtype=dtype,generator=generator)*(dt/4)**.5
            for k,level in enumerate((1,2,4)):
                for part in range(level):
                    noise=dw[part*(4//level):(part+1)*(4//level)].sum(0)
                    states[k],error=llg.step(states[k],dt/level,noise,method='heun')
                    max_raw[k]=max(max_raw[k],error.max().item())
                norm=(states[k].norm(dim=-1)-1).abs().max().item()
                max_norm[k]=max(max_norm[k],norm)
                if not torch.isfinite(states[k]).all() or norm>config['acceptance']['unit_norm_absolute']:
                    raise RuntimeError('Nonfinite spin or unit-norm failure')
            if step%stride==0: save(step//stride)
        for k,level in enumerate((1,2,4)):
            group=h.create_group(f'observables_dt_div_{level}')
            for key in all_obs[k][0]: group.create_dataset(key,data=np.stack([x[key] for x in all_obs[k]]))
        h.create_dataset('rng_state_final',data=generator.get_state().cpu().numpy())
    elapsed=time.perf_counter()-start
    differences=[(states[k]-states[-1]).square().mean().sqrt().item() for k in (0,1)]
    report={'system':args.system,'numerical_run_status':'COMPLETED_FINITE_UNIT_SPINS',
            'scientific_status':'NOT_CERTIFIED','production_enabled':False,
            'certification_holds':row['certification_holds'],
            'duration_ps':n['duration_ps'],'dt_fs':[n['dt_fs']/x for x in (1,2,4)],
            'wall_seconds':elapsed,'max_stage_norm_error':max_raw,'max_final_norm_error':max_norm,
            'coupled_final_spin_rms_vs_finest':differences,
            'note':'Small coupled ensemble diagnoses integration, not distributional equivalence or paper observables.',
            'source_hashes_unchanged':verify_sources()==manifest}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
