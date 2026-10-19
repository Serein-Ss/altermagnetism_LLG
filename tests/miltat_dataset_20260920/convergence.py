"""Full-dipole 2D step refinement with matched Brownian paths, stored in tests/."""
import argparse
import hashlib
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
from .topology import triangles, charge

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def coupled_step(llg, states, fine_dt, increments):
    """All three levels receive the SAME Wiener path, not merely the same RNG seed."""
    if len(increments) != 4 or len(states) != 3:
        raise ValueError('Expected four fine increments and three refinement levels')
    coarse, middle, fine = states
    coarse, _ = llg.step(coarse, fine_dt*4, sum(increments), method='heun')
    for k in [0, 2]:
        middle, _ = llg.step(middle, fine_dt*2, increments[k]+increments[k+1], method='heun')
    for dw in increments:
        fine, _ = llg.step(fine, fine_dt, dw, method='heun')
    return [coarse, middle, fine]


def run(index, device, smoke=False):
    if not smoke or device == 'cuda': allocation(device)
    torch.set_num_threads(1)
    cfg = yaml.safe_load((HERE/'convergence.yaml').read_text())
    source = yaml.safe_load((ROOT/cfg['source_config']).read_text())
    source['systems']['miltat']['physics'].update(temperature_K=cfg['temperature_K'], alpha=cfg['alpha'])
    if smoke: cfg.update(duration_ps=.01, save_ps=.0025)
    out = HERE/('smoke' if smoke else 'convergence')/f'branch_{index}'
    out.mkdir(parents=True, exist_ok=False)
    model, reduced, scales = build(source, 'miltat', shape=(6,4) if smoke else None)
    if smoke:
        s = model.uniform([1,1], batch=1, device=device, dtype=torch.float64)
    else:
        ck = torch.load(ROOT/cfg['initial_checkpoint'], map_location='cpu', weights_only=False)
        s = ck['spins'].to(device=device, dtype=torch.float64)
        assert tuple(s.shape[1:3]) == tuple(model.shape)
    pos = (model.positions(s)*reduced['a_nm']).cpu().numpy()
    tri = triangles(pos)
    seed = cfg['branch_seeds'][index]
    rng = torch.Generator(device=device).manual_seed(seed)
    coarse_dt = cfg['coarse_dt_fs']*1e-15/scales['time_s']
    steps = round(cfg['duration_ps']*1000/cfg['coarse_dt_fs'])
    stride = round(cfg['save_ps']*1000/cfg['coarse_dt_fs'])
    assert stride > 0 and steps % stride == 0
    llg = ReducedLLG(model, alpha=reduced['alpha'], theta=reduced['theta'], equation_convention='gilbert')
    states = [s.clone() for _ in range(3)]; records = []; started = time.perf_counter()
    with torch.inference_mode(), h5py.File(out/'paths.h5', 'x') as h:
        h.attrs['complete'] = False; h.attrs['config_json'] = json.dumps(cfg)
        h.attrs['physical_source_json'] = json.dumps(source['systems']['miltat'])
        h.attrs['scales_json'] = json.dumps(scales); h.attrs['reduced_json'] = json.dumps(reduced)
        h.attrs['noise_seed'] = seed
        if not smoke: h.attrs['initial_sha256'] = hashlib.sha256((ROOT/cfg['initial_checkpoint']).read_bytes()).hexdigest()
        h.create_dataset('positions_nm', data=pos)
        h.create_dataset('time_ps', data=np.arange(steps//stride+1)*cfg['save_ps'])
        d = h.create_dataset('spins', (3, steps//stride+1, *s.shape[1:]), dtype='f4',
                             chunks=(1,1,*s.shape[1:]), compression='lzf')
        def save(frame):
            row = dict(time_ps=frame*cfg['save_ps'], levels=[])
            for k, state in enumerate(states):
                array = state[0].cpu().numpy(); diag = charge(array, tri)
                if diag['stored_norm_error'] > 1e-10: raise RuntimeError('Double precision spin norm failure')
                d[k, frame] = array
                core = array[...,2] >= .5
                center = pos[core,:2].mean(0).tolist() if core.any() else None
                row['levels'].append(dict(dt_fs=cfg['coarse_dt_fs']/2**k, **diag,
                    energy_per_spin=float(model.energy(state).item()/(state.numel()/3)),
                    mean_sz=float(array[...,2].mean()), core_sites=int(core.sum()),
                    all_threshold_sites_center_nm=center,
                    warning='threshold center is not largest-component tracking'))
            row['spin_rmse_coarse_fine'] = float((states[0]-states[2]).square().mean().sqrt())
            row['spin_rmse_middle_fine'] = float((states[1]-states[2]).square().mean().sqrt())
            records.append(row); h.attrs['frames_written'] = frame+1; h.flush()
        save(0)
        for step in range(1, steps+1):
            dw = [llg.noise_increment(states[2], coarse_dt/4, rng) for _ in range(4)]
            states = coupled_step(llg, states, coarse_dt/4, dw)
            if step % stride == 0:
                save(step//stride)
                if step//stride % 10 == 0: print(index, step//stride, time.perf_counter()-started, flush=True)
        h.attrs['complete'] = True
    dump(out/'metrics.json', dict(rows=records, seconds=time.perf_counter()-started, smoke=smoke,
         config=cfg, gpu=torch.cuda.get_device_name() if device=='cuda' else None,
         scientific_certification=False, note='Paired strong-error diagnostic from one fixed parent; no equivalence claim'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--index', type=int, choices=range(4), default=0)
    p.add_argument('--device', choices=['cpu','cuda'], required=True); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); run(a.index, a.device, a.smoke)
