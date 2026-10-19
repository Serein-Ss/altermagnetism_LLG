"""Full-size CPU throughput, not a prepared physical reference trajectory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch
import yaml
from tests.literature_preflight_20260917.models import build
from scripts.core.reduced_llg import ReducedLLG

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    cfg=yaml.safe_load((HERE/'config.yaml').read_text())
    if not args.smoke and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Full-size benchmark requires a scheduled CPU allocation')
    torch.set_num_interop_threads(1)
    original=yaml.safe_load((ROOT/cfg['source']).read_text())
    row=original['systems'][cfg['system']]
    row['physics']['temperature_K']=cfg['temperature_K']
    row['physics']['alpha']=cfg['alpha']
    threads=cfg['threads']
    if args.smoke:
        threads=[1];cfg.update(warmup_steps=1,steps_per_repeat=4,repeats=2,save_every_steps=2)
    elif int(os.environ.get('SLURM_CPUS_PER_TASK','0'))<max(threads):
        raise RuntimeError('Insufficient requested CPUs')
    out=HERE/('smoke' if args.smoke else 'results')
    out.mkdir(exist_ok=True)
    results=[]
    with torch.inference_mode():
        for nthreads in threads:
            torch.set_num_threads(nthreads)
            begin=time.perf_counter()
            model,reduced,scales=build(original,cfg['system'],shape=(6,4) if args.smoke else None)
            s=model.uniform([1]*model.basis,batch=1,device='cpu',dtype=torch.float64)
            s=model.initial_texture(s,**{k:row['preparation'][k] for k in ['radius_nm','width_nm']})
            initial=s.clone()
            dt=cfg['dt_fs']*1e-15/scales['time_s']
            llg=ReducedLLG(model,alpha=reduced['alpha'],theta=reduced['theta'],equation_convention='gilbert')
            rng=torch.Generator(device='cpu').manual_seed(cfg['seed'])
            for _ in range(cfg['warmup_steps']):
                s,_=llg.step(s,dt,llg.noise_increment(s,dt,rng),method='heun')
            startup=time.perf_counter()-begin
            # Restart identical trial state and noise for each thread count.
            s=initial.clone();rng.manual_seed(cfg['seed'])
            total=cfg['steps_per_repeat']*cfg['repeats']
            stride=cfg['save_every_steps']
            assert total%stride==0
            elapsed=[];max_norm=0.;max_raw=0.
            with h5py.File(out/f'threads_{nthreads}.h5','x') as h:
                h.attrs['scope']='unrelaxed_texture_benchmark_not_training_data'
                h.attrs['device']='cpu';h.attrs['threads']=nthreads
                h.create_dataset('time_ps',data=np.arange(total//stride+1)*stride*cfg['dt_fs']/1000)
                ds=h.create_dataset('spins',shape=(total//stride+1,*s.shape),dtype='f8',
                                    chunks=(1,*s.shape),compression='lzf')
                energy=h.create_dataset('energy_reduced',shape=(total//stride+1,),dtype='f8')
                mz=h.create_dataset('mean_sz',shape=(total//stride+1,),dtype='f8')
                ds[0]=s.numpy();energy[0]=model.energy(s).item();mz[0]=s[...,2].mean().item()
                for rep in range(cfg['repeats']):
                    start=time.perf_counter()
                    for i in range(cfg['steps_per_repeat']):
                        s,error=llg.step(s,dt,llg.noise_increment(s,dt,rng),method='heun')
                        max_raw=max(max_raw,error.max().item())
                        err=(s.norm(dim=-1)-1).abs().max().item();max_norm=max(max_norm,err)
                        if not torch.isfinite(s).all() or err>1e-10:
                            raise RuntimeError('Numerical integrity failure during benchmark')
                        step=rep*cfg['steps_per_repeat']+i+1
                        if step%stride==0:
                            k=step//stride;ds[k]=s.numpy()
                            energy[k]=model.energy(s).item();mz[k]=s[...,2].mean().item();h.flush()
                    elapsed.append(time.perf_counter()-start)
                    print(json.dumps(dict(threads=nthreads,repeat=rep,seconds=elapsed[-1])),flush=True)
                h.attrs['complete']=True
            ps=cfg['steps_per_repeat']*cfg['dt_fs']/1000
            rates=np.array(elapsed)/ps
            results.append(dict(threads=nthreads,spins=s.numel()//3,dtype=str(s.dtype),
                startup_and_warmup_seconds=startup,repeat_physical_ps=ps,repeat_seconds=elapsed,
                seconds_per_ps=float(rates.mean()),ten_ns_hours=float(rates.mean()*10000/3600),
                ten_ns_hours_repeat_range=(rates*10000/3600).tolist(),
                max_norm_error=max_norm,max_stage_error=max_raw))
            report=dict(scope='CPU_SPEED_ONLY_NOT_DISTRIBUTION_CERTIFIED',device='cpu',
                config=cfg,measurements=results,torch=torch.__version__,
                allocation_cpus=os.environ.get('SLURM_CPUS_PER_TASK'),
                timing_includes='noise, Heun, per-step integrity, HDF5 full float64 frames and energy every 0.1 ps',
                extrapolation_excludes='queue, preparation, moving window, stability fixes, long-run filesystem variation',
                hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [HERE/'run.py',HERE/'config.yaml',ROOT/cfg['source'],
                                  ROOT/'tests/literature_preflight_20260917/models.py',ROOT/'scripts/core/reduced_llg.py']})
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(results[-1]),flush=True)


if __name__=='__main__':main()
