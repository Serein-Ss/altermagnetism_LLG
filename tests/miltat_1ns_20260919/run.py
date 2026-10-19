"""One full-dipole thermal demonstration; not a diffusion certification."""
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
from scripts.core.reduced_llg import ReducedLLG
from tests.literature_preflight_20260917.models import build
from tests.physical_audit_20260917.cold import strong_rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True, choices=['cpu', 'cuda'])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not args.smoke and (args.device != 'cuda' or not os.environ.get('SLURM_JOB_ID')):
        raise RuntimeError('Full run requires scheduled CUDA')
    torch.set_num_threads(1)
    cfg = yaml.safe_load((HERE/'config.yaml').read_text())
    if args.smoke:
        cfg.update(preparation_ps=0.01, duration_ps=0.02, save_ps=0.005)
    source = yaml.safe_load((ROOT/cfg['source']).read_text())
    row = source['systems']['miltat']
    row['physics'].update(temperature_K=cfg['temperature_K'], alpha=cfg['alpha'])
    out = HERE/('smoke' if args.smoke else 'results')
    out.mkdir(exist_ok=False)
    report = dict(config=cfg, physical_source=row, device=args.device,
                  scope='single_trajectory_demonstration_not_equilibrated_distribution_certification',
                  limitations=['fixed open window, not the paper moving window',
                               'finite zero-T preparation is not an equilibrium certificate',
                               '1 ns does not cover the full Fig.10 fitting range'], phases={})
    report['hashes'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [HERE/'run.py', HERE/'config.yaml', ROOT/cfg['source'],
                                  ROOT/'tests/literature_preflight_20260917/models.py',
                                  ROOT/'scripts/core/reduced_llg.py',
                                  ROOT/'tests/physical_audit_20260917/cold.py']}
    if args.device == 'cuda':
        report['gpu'] = torch.cuda.get_device_name()
    with torch.inference_mode():
        model, reduced, scales = build(source, 'miltat', shape=(6, 4) if args.smoke else None)
        s = model.uniform([1]*model.basis, batch=1, device=args.device, dtype=torch.float64)
        s = model.initial_texture(s, **{k: row['preparation'][k] for k in ['radius_nm', 'width_nm']})
        s = strong_rotation(s, model.shape, cfg)
        positions = (model.positions(s)*reduced['a_nm']).cpu().numpy()
        dt = cfg['dt_fs']*1e-15/scales['time_s']
        stride = round(cfg['save_ps']*1000/cfg['dt_fs'])
        rng = torch.Generator(device=args.device).manual_seed(cfg['seed'])
        for phase, duration, theta in [('preparation', cfg['preparation_ps'], 0.),
                                       ('thermal', cfg['duration_ps'], reduced['theta'])]:
            llg = ReducedLLG(model, alpha=reduced['alpha'], theta=theta, equation_convention='gilbert')
            steps = round(duration*1000/cfg['dt_fs'])
            assert steps % stride == 0
            started = time.perf_counter()
            maximum = 0.
            with h5py.File(out/f'{phase}.h5', 'x') as h:
                h.attrs['complete'] = False
                h.attrs['temperature_K'] = 0. if phase == 'preparation' else cfg['temperature_K']
                h.attrs['config_json'] = json.dumps(cfg)
                h.attrs['scope'] = report['scope']
                h.attrs['boundary'] = 'fixed_open_full_dipole'
                h.attrs['dt_fs'] = cfg['dt_fs']
                h.attrs['method'] = 'heun'
                h.create_dataset('positions_nm', data=positions)
                h.create_dataset('time_ps', data=np.arange(steps//stride+1)*cfg['save_ps'])
                spins = h.create_dataset('spins', (steps//stride+1, *s.shape), dtype='f4', chunks=(1, *s.shape))
                energy = h.create_dataset('energy_reduced', (steps//stride+1,), dtype='f8')
                norm = h.create_dataset('norm_error_float64', (steps//stride+1,), dtype='f8')
                def save(index):
                    err = (s.norm(dim=-1)-1).abs().max().item()
                    if not torch.isfinite(s).all().item() or err > 1e-10:
                        raise RuntimeError('Nonfinite state or spin norm failure')
                    spins[index] = s.cpu().numpy()
                    energy[index] = model.energy(s).item()
                    norm[index] = err
                    h.attrs['frames_written'] = index+1
                    h.flush()
                    return err
                maximum = max(maximum, save(0))
                for step in range(1, steps+1):
                    dw = llg.noise_increment(s, dt, rng) if theta else torch.zeros_like(s)
                    s, _ = llg.step(s, dt, dw, method='heun')
                    if step % stride == 0:
                        maximum = max(maximum, save(step//stride))
                        if step//stride % 10 == 0 or step == steps:
                            elapsed = time.perf_counter()-started
                            print(json.dumps(dict(phase=phase, physical_ps=step*cfg['dt_fs']/1000,
                                                  elapsed_seconds=elapsed,
                                                  estimated_remaining_seconds=elapsed*(steps-step)/step)), flush=True)
                h.attrs['complete'] = True
                h.attrs['max_saved_norm_error_float64'] = maximum
                if phase == 'preparation':
                    last = np.asarray(energy[-min(11, len(energy)):])
                    report['preparation_final_energy_span_per_spin'] = float(np.ptp(last)/(s.numel()/3))
            report['phases'][phase] = dict(seconds=time.perf_counter()-started, duration_ps=duration,
                                           max_saved_norm_error_float64=maximum)
            torch.save(dict(spins=s.cpu(), rng=rng.get_state(), config=cfg, phase=phase), out/f'{phase}_final.pt')
            (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
