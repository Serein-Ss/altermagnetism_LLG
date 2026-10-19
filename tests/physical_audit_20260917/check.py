"""Independent limiting-case checks and cross-integrator diagnostics on CUDA."""
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

BASE = Path(__file__).resolve().parent
PILOT = BASE.parent / 'literature_preflight_20260917'


class ConstantField:
    def __init__(self, hz): self.hz = hz
    def field(self, s):
        h = torch.zeros_like(s); h[..., 2] = self.hz
        return h


def main():
    p = argparse.ArgumentParser(); p.add_argument('--system', required=True, choices=['bauer', 'miltat', 'hirst']); args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Slurm CUDA allocation required')
    torch.set_num_threads(1)
    cfg = yaml.safe_load((PILOT / 'config.yaml').read_text())
    settings = yaml.safe_load((BASE / 'settings.yaml').read_text())
    out = BASE / 'results' / args.system; out.mkdir(parents=True, exist_ok=True)
    settings_c = settings['coupled_integrators']
    model, r, scale = build(cfg, args.system, shape=settings_c['shapes'][args.system])
    convention = 'bauer_ll' if args.system == 'bauer' else 'gilbert'
    rng = torch.Generator(device='cuda').manual_seed(settings['seed'])
    dtype = torch.float64
    def normal(shape): return torch.randn(shape, device='cuda', dtype=dtype, generator=rng)
    # Closed-form deterministic single-spin solution in unit z field.
    s = torch.tensor([[[1., 0., 0.]]], device='cuda', dtype=dtype)
    llg = ReducedLLG(ConstantField(1.), alpha=r['alpha'], theta=0., equation_convention=convention)
    dt = .001
    for _ in range(1000): s, _ = llg.step(s, dt, torch.zeros_like(s), method='heun_projected_predictor')
    pref = 1. if convention == 'bauer_ll' else 1 / (1 + r['alpha'] ** 2)
    z = np.tanh(r['alpha'] * pref)
    exact = s.new_tensor([np.sqrt(1 - z*z)*np.cos(pref), np.sqrt(1 - z*z)*np.sin(pref), z])
    analytic_error = float((s - exact).abs().max())
    # Free thermal spin: <sz(t)> = exp(-2 D t), not a fitted equilibrium result.
    free = settings['free_spin']; count = free['samples']
    s = torch.zeros((count, 1, 3), device='cuda', dtype=dtype); s[..., 2] = 1.
    llg = ReducedLLG(ConstantField(0.), alpha=r['alpha'], theta=r['theta'], equation_convention=convention)
    D = r['alpha'] * r['theta'] * pref
    step_dt = free['reduced_dt']
    dw = normal(s.shape) * np.sqrt(step_dt)
    inc = llg.increment(s, step_dt, dw)
    variance = inc[:, 0, :2].var(0).cpu().numpy()
    expected_var = 2 * D * step_dt
    for _ in range(free['steps']):
        s, _ = llg.step(s, step_dt, normal(s.shape)*np.sqrt(step_dt), method='heun_projected_predictor')
    observed = float(s[..., 2].mean()); se = float(s[..., 2].std() / np.sqrt(count))
    expected = float(np.exp(-2 * D * free['steps'] * step_dt))
    thermal = dict(mean_z=observed, analytical_mean_z=expected, standard_error=se,
                   z_score=abs(observed - expected)/max(se, 1e-30), samples=count,
                   variance_xy=variance.tolist(), expected_variance=expected_var,
                   variance_relative_error=(variance/expected_var-1).tolist(),
                   passed=bool(abs(observed - expected) <= free['confidence_sigma']*se + 2e-5
                          and np.max(np.abs(variance/expected_var-1)) < free['confidence_sigma']*np.sqrt(2/(count-1))))
    # General Hamiltonian energy balance at T=0, plus energy descent integration.
    s = torch.nn.functional.normalize(normal((2, *model.shape, model.basis, 3)), dim=-1)
    cold = ReducedLLG(model, alpha=r['alpha'], theta=0., equation_convention=convention)
    field = model.field(s)
    velocity = cold.increment(s, 1., torch.zeros_like(s))
    dE = -(field * velocity).flatten(1).sum(1)
    exact_dE = -r['alpha']*pref*torch.linalg.cross(s, field).square().flatten(1).sum(1)
    balance = float((dE-exact_dE).abs().max()/exact_dE.abs().max().clamp_min(1))
    dt = cfg['systems'][args.system]['numerics']['dt_fs']*1e-15/scale['time_s']
    energy = [model.energy(s).cpu().tolist()]
    for _ in range(128):
        s, _ = cold.step(s, dt/4, torch.zeros_like(s), method='heun_projected_predictor')
        energy.append(model.energy(s).cpu().tolist())
    rise = float(np.diff(np.array(energy), axis=0).max())
    # Matched Wiener increments; independent implicit midpoint reference.
    llg = ReducedLLG(model, alpha=r['alpha'], theta=r['theta'], equation_convention=convention)
    batch = settings_c['batch']
    anchor = model.ground_state(batch=batch,device='cuda',dtype=dtype) if args.system == 'hirst' else model.uniform([1]*model.basis,batch=batch,device='cuda',dtype=dtype)
    anchor = torch.nn.functional.normalize(anchor + .1*normal(anchor.shape), dim=-1)
    states = [anchor.clone() for _ in range(4)]
    for step in range(settings_c['coarse_steps']):
        dw = [normal(anchor.shape)*np.sqrt(dt/4) for _ in range(4)]
        for k, level in enumerate([1,2,4]):
            for j in range(level):
                inc = sum(dw[j*(4//level):(j+1)*(4//level)])
                states[k], _ = llg.step(states[k], dt/level, inc, method='heun_projected_predictor')
        for inc in dw:
            states[3], _ = llg.step(states[3], dt/4, inc, method='geometric_midpoint',
                                   tolerance=settings_c['midpoint_tolerance'])
    rms = [float((s-states[3]).square().mean().sqrt()) for s in states[:3]]
    differences = [(model.energy(s)-model.energy(states[3])).div(s[0,...,0].numel()).cpu().numpy() for s in states[:3]]
    comparison = dict(shape=list(model.shape), batch=batch, duration_ps=settings_c['coarse_steps']*dt*scale['time_s']*1e12,
                      dt_fs=[dt*scale['time_s']*1e15/i for i in (1,2,4)], rms_vs_midpoint=rms,
                      energy_bias_per_spin=[float(d.mean()) for d in differences],
                      energy_bias_se=[float(d.std(ddof=1)/np.sqrt(batch)) for d in differences],
                      refinement_reduces_rms=rms[-1]<rms[0], note='Numerical consistency on smaller diagnostic geometries, not full-domain distribution certification')
    result = dict(system=args.system, constant_field_error=analytic_error, thermal_diffusion=thermal,
                  energy_balance_relative_error=balance, largest_zero_T_energy_rise=rise,
                  integrator_comparison=comparison, job_id=os.environ['SLURM_JOB_ID'],
                  basic_checks_passed=bool(analytic_error<1e-5 and thermal['passed'] and balance<1e-10 and rise<=1e-9),
                  scientific_status='PARTIAL_VALIDATION_NOT_RELEASED', production_enabled=False)
    (out / 'checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if args.system == 'miltat':
        # Does the actually prepared texture survive zero-temperature relaxation?
        full, r, scale = build(cfg, 'miltat')
        with h5py.File(PILOT/'results/miltat/trajectory.h5') as f:
            s = torch.tensor(f['spins'][-1, :1], device='cuda', dtype=dtype)
        relax = settings['texture_relaxation']; dt_fs = relax['dt_fs']
        cold = ReducedLLG(full, alpha=r['alpha'], theta=0.)
        steps = round(relax['duration_ps']*1000/dt_fs); stride = round(relax['save_ps']*1000/dt_fs)
        with h5py.File(out/'texture_relaxation.h5', 'w') as h:
            frames = h.create_dataset('spins', (steps//stride+1, *s.shape), dtype='f8', compression='lzf')
            h.create_dataset('time_ps', data=np.arange(steps//stride+1)*relax['save_ps'])
            frames[0] = s.cpu().numpy()
            for step in range(1, steps+1):
                s, _ = cold.step(s, dt_fs*1e-15/scale['time_s'], torch.zeros_like(s), method='heun_projected_predictor')
                if step % stride == 0: frames[step//stride] = s.cpu().numpy()
            h.attrs['protocol'] = 'Diagnostic T=0 relaxation from old pilot final state; NOT training data'
    if not result['basic_checks_passed']: raise RuntimeError('Basic physical limiting-case check failed')


if __name__ == '__main__': main()
