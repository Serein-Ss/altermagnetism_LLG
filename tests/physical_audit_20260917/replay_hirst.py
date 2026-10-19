"""Paired replay of the frozen Hirst pilot, isolating predictor normalization."""
import json
import os
from pathlib import Path
import h5py
import numpy as np
import torch
import yaml
from scripts.core.reduced_llg import ReducedLLG
from tests.literature_preflight_20260917.models import build


def main():
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available(): raise RuntimeError('Slurm CUDA required')
    torch.set_num_threads(1)
    base=Path(__file__).resolve().parent
    pilot=base.parent/'literature_preflight_20260917'
    cfg=yaml.safe_load((pilot/'config.yaml').read_text())
    meta=json.loads((pilot/'results/hirst/metadata.json').read_text())
    cell,r,scale=build(cfg,'hirst'); n=cfg['systems']['hirst']['numerics']
    dt=n['dt_fs']*1e-15/scale['time_s']; steps=round(n['duration_ps']*1000/n['dt_fs'])
    stride=round(n['save_ps']*1000/n['dt_fs'])
    old=cell.ground_state(batch=n['batch'],device='cuda',dtype=torch.float64); new=old.clone()
    llg=ReducedLLG(cell,alpha=r['alpha'],theta=r['theta'])
    rng=torch.Generator(device='cuda').manual_seed(meta['seed'])
    out=base/'results/hirst'; replay_error=[]; paired_rms=[]; neel_old=[]; neel_new=[]
    with h5py.File(pilot/'results/hirst/trajectory.h5') as original, h5py.File(out/'paper_heun_replay.h5','w') as dest:
        frames=dest.create_dataset('spins',shape=original['spins'].shape,dtype='f8',compression='lzf')
        dest.create_dataset('time_ps',data=original['time_ps'][:]); frames[0]=new.cpu().numpy()
        def record(i):
            replay_error.append(float(np.abs(old.cpu().numpy()-original['spins'][i]).max()))
            paired_rms.append(float((new-old).square().mean().sqrt()))
            neel_old.append(cell.order_parameters(old)['neel'].cpu().tolist())
            neel_new.append(cell.order_parameters(new)['neel'].cpu().tolist())
            frames[i]=new.cpu().numpy()
        record(0)
        for step in range(1,steps+1):
            dw=torch.randn((4,*old.shape),device='cuda',dtype=torch.float64,generator=rng)*(dt/4)**.5
            for noise in dw:
                old,_=llg.step(old,dt/4,noise,method='heun')
                new,_=llg.step(new,dt/4,noise,method='heun_projected_predictor')
            if step%stride==0: record(step//stride)
        exact_rng=bool(np.array_equal(rng.get_state().cpu().numpy(),original['rng_state_final'][:]))
        dest.attrs['scope']='Test-only paired replay, not a certified dataset'
    result=dict(original_replay_max_error=max(replay_error),identical_final_rng_state=exact_rng,
                paired_spin_rms_by_frame=paired_rms,neel_old=neel_old,neel_paper_heun=neel_new,
                final_neel_max_difference=float(np.abs(np.array(neel_old[-1])-np.array(neel_new[-1])).max()),
                original_seed=meta['seed'],job=os.environ['SLURM_JOB_ID'],production_enabled=False)
    (out/'paper_heun_replay.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    assert exact_rng and max(replay_error)<1e-10, 'Replay not matched; cannot isolate integrator change'


if __name__=='__main__': main()
