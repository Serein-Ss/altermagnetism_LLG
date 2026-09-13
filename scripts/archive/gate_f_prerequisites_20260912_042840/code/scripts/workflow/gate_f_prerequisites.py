"""Bounded Gate-F prerequisite campaign. All computation is submitted to fat.

MC -> kernel ensembles -> static gate -> size calibrations -> fine-window
diagnostics -> report. A failed gate cancels dependents, not silent COMPLETED
skips. Reports never certify production or rewrite the historical failed gate.
"""
import argparse
from datetime import datetime, timezone
import json
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import h5py
import numpy as np
import torch
import yaml
from scipy.stats import t as student
from scripts.core.literature_config import load_runtime, sha256
from scripts.core.reduced_llg import ReducedLLG
from scripts.core.streaming_llg import simulate_resumable
from scripts.literature.gomonay_2024.model import DoubleLayer
from scripts.generation.generate_zero_field_gomonay import run as run_llg, DRIVES
from scripts.validation.gomonay_static_reference import coupling_matrix, sample as sample_mc
from scripts.analysis.zero_field_equilibrium import diagnostics

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = [
    'scripts/workflow/gate_f_prerequisites.py', 'scripts/core/literature_config.py',
    'scripts/core/reduced_llg.py', 'scripts/core/cell_hamiltonian.py', 'scripts/core/streaming_llg.py',
    'scripts/literature/gomonay_2024/model.py', 'scripts/generation/generate_zero_field_gomonay.py',
    'scripts/validation/gomonay_static_reference.py', 'scripts/analysis/zero_field_equilibrium.py',
    'conf/literature/gomonay_2024.yaml', 'conf/gate_f/prerequisites.yaml',
    'slurm/gate_f_prerequisites.sbatch', 'scripts/tests/test_gate_f_prerequisites.py',
]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def finite(obj):
        if isinstance(obj, dict):
            return {k: finite(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [finite(v) for v in obj]
        if isinstance(obj, np.generic):
            obj = obj.item()
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj
    with path.open('x') as h:
        json.dump(finite(value), h, indent=2, allow_nan=False)


def configuration(design, run_id, material, size, theta, duration, dt, save_dt, chains, method, seed):
    return dict(run_id=run_id, role='equilibrium_diagnostic', production_enabled=False,
                material_config=str(material), drives={k: 0 for k in DRIVES}, orientation='100',
                shape=[size, size], periodic=[True, True], batch=chains,
                initializations=(['positive', 'negative', 'random', 'random']*(chains//4)),
                initial_seed=seed+100000, noise_seeds=list(range(seed, seed+chains)), seed=seed,
                alpha=design['alpha'], theta=theta, method=method, dt=dt,
                steps=round(duration/dt), save_every=round(save_dt/dt), checkpoint_every=10000)


def observations(energy, neel, sites):
    return dict(energy_per_spin=energy/sites, neel_norm=np.linalg.norm(neel, axis=-1),
                neel_z_squared=neel[..., 2]**2, signed_neel_z=neel[..., 2])


def mean_interval(x, alpha):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError('at least four finite independent chain means required')
    half = student.ppf(1-alpha/2, len(x)-1)*x.std(ddof=1)/math.sqrt(len(x))
    return [float(x.mean()-half), float(x.mean()+half)]


def summarize(obs, spacing, settings):
    result = {}
    alpha = (1-settings['confidence'])/settings['multiplicity_family_bound']
    for name, x in obs.items():
        x = x[:, int(x.shape[1]*settings['burn_fraction']):]
        d = diagnostics(x, spacing)
        half = x.shape[1]//2
        drift = x[:, -half:].mean(1)-x[:, :half].mean(1)
        interval = mean_interval(drift, alpha)
        passed = (np.isfinite(d['rank_folded_rhat']) and d['rank_folded_rhat'] <= settings['rhat_max']
                  and d['bulk_ess'] >= settings['bulk_ess_min'] and d['tail_ess'] >= settings['tail_ess_min']
                  and max(abs(v) for v in interval) <= settings['absolute_band'])
        result[name] = dict(**d, drift_mean=float(drift.mean()), drift_ci=interval, passed=bool(passed))
    return result


def equivalent(left, right, settings):
    x, y = np.asarray(left), np.asarray(right)
    vx, vy = x.var(ddof=1)/len(x), y.var(ddof=1)/len(y)
    variance = vx+vy
    df = variance**2/(vx*vx/(len(x)-1)+vy*vy/(len(y)-1)) if variance else len(x)+len(y)-2
    alpha = (1-settings['confidence'])/settings['multiplicity_family_bound']
    half = student.ppf(1-alpha/2, df)*math.sqrt(variance)
    delta = x.mean()-y.mean()
    ci = [float(delta-half), float(delta+half)]
    return dict(difference=float(delta), ci=ci, passed=max(abs(v) for v in ci) <= settings['absolute_band'])


def load_llg(path, size):
    with h5py.File(path) as h:
        if not h.attrs['complete']:
            raise ValueError('incomplete trajectory')
        obs = observations(h['energy'][:].T, h['neel'][:].transpose(1, 0, 2), 2*size*size)
        spacing = float(h['time'][1]-h['time'][0])
        norm = float(h['max_norm_error'][:].max())
    return obs, spacing, norm


def mc_worker(m, index):
    d = m['design']; theta = d['candidate_theta'][index]
    r = load_runtime(m['material']).reduced
    matrix = coupling_matrix(d['kernel']['size'], r)
    k = d['kernel']; states, rates = [], []
    for chain in range(k['chains']):
        x, rate = sample_mc(matrix, r['K_DW'], theta, k['mc_sweeps'], k['mc_stride'],
                            d['seed']+index*100+chain, chain % 4)
        states.append(x); rates.append(rate)
    states = np.asarray(states)
    energy = -.5*np.einsum('ctia,ij,ctja->ct', states, matrix, states)-r['K_DW']*(states[..., 2]**2).sum(-1)
    signs = np.tile([1., -1.], d['kernel']['size']**2)
    neel = (states*signs[None, None, :, None]).mean(2)
    path = Path(m['raw'])/f'mc_{index}.npz'
    if path.exists():
        raise FileExistsError(path)
    temporary = Path(str(path)+'.partial')
    with temporary.open('xb') as h:
        np.savez_compressed(h, spins=states, energy=energy, neel=neel,
                            acceptance=rates, metadata=json.dumps(dict(scope='MC_sweeps_not_physical_time', theta=theta)))
    temporary.rename(path)


def kernel_gate(m):
    d, results = m['design'], []
    settings = d['statistics']; k = d['kernel']
    for i, theta in enumerate(d['candidate_theta']):
        with np.load(Path(m['raw'])/f'mc_{i}.npz') as mc:
            ref = summarize(observations(mc['energy'], mc['neel'], 2*k['size']**2), k['mc_stride'], settings)
        methods = []
        for j, method in enumerate(k['methods']):
            obs, spacing, norm = load_llg(Path(m['raw'])/f'kernel_{i*4+j}.h5', k['size'])
            stats = summarize(obs, spacing, settings)
            comparisons = {name: equivalent(stats[name]['chain_means'], ref[name]['chain_means'], settings)
                           for name in settings['primary']}
            passed = norm <= 1e-10 and all(stats[name]['passed'] and ref[name]['passed'] and comparisons[name]['passed'] for name in settings['primary'])
            methods.append(dict(method=method, statistics=stats, comparisons=comparisons, norm_error=norm, passed=passed))
        results.append(dict(theta=theta, mc=ref, methods=methods, passed=all(row['passed'] for row in methods)))
    eligible = [row['theta'] for row in results if row['passed']]
    chosen = [eligible[0], eligible[(len(eligible)-1)//2], eligible[-1]] if len(eligible) >= 3 else []
    dump(Path(m['output'])/'kernel_gate.json', dict(status='pass' if chosen else 'inconclusive',
         scope='preliminary_static_kernel_for_calibration_only', production_enabled=False,
         alpha=d['alpha'], provisional_theta=chosen, results=results,
         limitation='not full P1/P2; no final Gate-F parameter freeze'))
    if not chosen:
        raise SystemExit(3)


def calibration_config(m, index, fine=False):
    gate = json.loads((Path(m['output'])/'kernel_gate.json').read_text())
    if gate['status'] != 'pass':
        raise ValueError('kernel gate has not passed')
    size = m['design']['calibration']['sizes'][index//3]
    theta = gate['provisional_theta'][index % 3]
    p = m['design']['fine_window' if fine else 'calibration']
    return configuration(m['design'], m['run_id']+f'_calibration_{index}', m['material'], size, theta,
                         p['duration'], p['dt'], p['save_dt'], 12 if fine else p['chains'],
                         'heun_projected_predictor', m['design']['seed']+20000+index*100+(10000 if fine else 0))


def fine_worker(m, index, half=False):
    c = calibration_config(m, index, True)
    if half:
        c['dt'] /= 2
        c['steps'] *= 2
        c['save_every'] *= 2
        c['noise_seeds'] = [seed+500000 for seed in c['noise_seeds']]
    cell = DoubleLayer(load_runtime(m['material']).reduced, c['shape'], wall=True)
    llg = ReducedLLG(cell, alpha=c['alpha'], theta=c['theta'])
    s = cell.ground_state(batch=12)
    g = torch.Generator().manual_seed(c['initial_seed'])
    perturb = torch.randn(s[:4].shape, dtype=s.dtype, generator=g)*m['design']['fine_window']['perturbation_sigma']
    s[:4] += perturb; s[:4] /= s[:4].norm(dim=-1, keepdim=True)
    s[4:8] = torch.randn(s[4:8].shape, dtype=s.dtype, generator=g)
    s[4:8] /= s[4:8].norm(dim=-1, keepdim=True)
    # Independent source chains, not adjacent frames. NOT called equilibrium until reviewed.
    with h5py.File(Path(m['raw'])/f'calibration_{index}.h5') as h:
        if not h.attrs['complete']:
            raise ValueError('incomplete source calibration')
        s[8:] = torch.from_numpy(h['spins'][-1, [0, 1, 4, 5]])
    signs = s.new_tensor([1., -1.])[None, None, None, :, None]
    def observe(x):
        n = (x*signs).mean(3)
        return dict(neel=n.mean((1, 2)), neel_spatial_std=n.std((1, 2), unbiased=False))
    c['role'] = 'fine_window_diagnostic_not_training'
    c['initializations'] = ['ground_perturbed']*4+['random_sphere']*4+['equilibrium_candidate']*4
    path = Path(m['raw'])/f'fine{"_half" if half else ""}_{index}.h5'
    _, done = simulate_resumable(path, s, llg, c, identity=dict(manifest=m['manifest_hash'],
                                source_sha256=sha256(Path(m['raw'])/f'calibration_{index}.h5')),
                                observer=observe)
    if not done:
        raise SystemExit(85)


def review(m):
    d, rows, initial_catalog = m['design'], [], []
    required = [Path(m['raw'])/f'{stage}_{index}.h5' for stage in ('calibration', 'fine', 'fine_half') for index in range(6)]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        dump(Path(m['output'])/'prerequisite_review.json', dict(status='inconclusive_missing_prerequisites',
             production_enabled=False, missing=missing, pending=d['unresolved_before_production']))
        raise SystemExit(3)
    for index in range(6):
        c = calibration_config(m, index)
        path = Path(m['raw'])/f'calibration_{index}.h5'
        obs, spacing, norm = load_llg(path, c['shape'][0])
        statistics = summarize(obs, spacing, d['statistics'])
        # Signed mixing is NOT required for the static kernel but IS reported for a global pool.
        stationary = all(statistics[k]['passed'] for k in d['statistics']['primary']) and norm <= 1e-10
        global_mixing = statistics['signed_neel_z']['passed']
        fine_path = Path(m['raw'])/f'fine_{index}.h5'
        with h5py.File(fine_path) as h:
            if not h.attrs['complete']:
                raise ValueError('incomplete fine-window run')
            time_grid, nz = h['time'][:], h['neel'][:, :, 2]
            initial_states = h['initial_state'][:]
            windows = {}
            for window in d['fine_window']['windows']:
                x = nz[time_grid <= window]
                windows[str(window)] = {str(stride): dict(saved_frames=len(x[::stride]),
                     crossed_zero_count=int(np.any(x[::stride][1:]*x[::stride][:-1] < 0, axis=0).sum()))
                     for stride in d['fine_window']['decimation']}
        with h5py.File(Path(m['raw'])/f'fine_half_{index}.h5') as h:
            if not h.attrs['complete']:
                raise ValueError('incomplete half-step diagnostic')
            other = h['neel'][:, :, 2]
            fine_norm = float(h['max_norm_error'][:].max())
        weak = {}
        for kind, start in [('ground_perturbed', 0), ('random_sphere', 4), ('equilibrium_candidate', 8)]:
            weak[kind] = equivalent(nz[-1, start:start+4], other[-1, start:start+4], d['statistics'])
        calibration_sha, fine_sha = sha256(path), sha256(fine_path)
        rows.append(dict(size=c['shape'][0], theta=c['theta'], statistics=statistics,
                         basin_even_stationarity=stationary, global_signed_mixing=global_mixing,
                         source_sha256=calibration_sha, fine_sha256=fine_sha, window_diagnostics=windows,
                         half_step_endpoint_weak_diagnostics=weak, half_step_norm_error=fine_norm,
                         weak_scope='independent noise, common initial states, four paths/type; not a high-power full-path certificate'))
        for initial_index, state in enumerate(initial_states):
            kind = ['ground_perturbed', 'random_sphere', 'equilibrium_candidate'][initial_index//4]
            local_index = initial_index % 4
            identity = f'{m["run_id"]}_condition_{index}_{kind}_{local_index}'
            source_chain = f'{m["run_id"]}_calibration_{index}_chain_{[0, 1, 4, 5][local_index]}' if kind == 'equilibrium_candidate' else None
            family = source_chain or identity
            state = np.ascontiguousarray(state.transpose(2, 0, 1, 3))
            candidate_path = Path(m['raw'])/'initial_candidates'/f'{identity}.npz'
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            with candidate_path.open('xb') as h:
                np.savez_compressed(h, initial=state)
            initial_catalog.append(dict(initial_id=identity, path=str(candidate_path), shape=list(state.shape),
                    dtype=str(state.dtype), spins_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                    file_sha256=sha256(candidate_path), initial_type=kind, source_family_id=family,
                    source_chain_id=source_chain, source_sha256=calibration_sha if source_chain else fine_sha,
                    split='train' if local_index < 2 else 'development', theta=c['theta'], alpha=c['alpha'],
                    size=c['shape'][0], physics_certified=False,
                    pool_global_stationarity_evidence=bool(stationary and global_mixing) if source_chain else None))
    dump(Path(m['output'])/'initial_candidates.json', dict(status='candidates_not_certified_not_training',
         production_enabled=False, initials=initial_catalog,
         note='72 candidate full initial states; no pool may be promoted merely because files exist'))
    dump(Path(m['output'])/'prerequisite_review.json', dict(status='measured_review_required',
         production_enabled=False, gate_f_contract_frozen=False, results=rows,
         pending=d['unresolved_before_production'],
         note='No full equilibrium pool, final theta/time selection or P1/P2 certificate is manufactured.'))


def worker(manifest, stage, index):
    m = json.loads(Path(manifest).read_text()); m['manifest_hash'] = sha256(manifest)
    if stage in m.get('execution_order', {}):
        index = m['execution_order'][stage][index]
    for relative, expected in m['files_sha256'].items():
        if sha256(Path(m['snapshot'])/relative) != expected:
            raise ValueError('frozen source changed: '+relative)
    if stage == 'mc':
        mc_worker(m, index)
    elif stage == 'kernel':
        config = Path(m['configs'])/f'kernel_{index}.yaml'
        if sha256(config) != m['configs_sha256'][config.name]:
            raise ValueError('kernel config changed')
        _, done = run_llg(config, Path(m['raw'])/f'kernel_{index}.h5', device='cpu')
        if not done:
            raise SystemExit(85)
    elif stage == 'gate':
        kernel_gate(m)
    elif stage == 'calibration':
        c = calibration_config(m, index)
        config = Path(m['configs'])/f'calibration_{index}.yaml'
        with config.open('x') as h:
            yaml.safe_dump(c, h)
        _, done = run_llg(config, Path(m['raw'])/f'calibration_{index}.h5', device='cpu')
        if not done:
            raise SystemExit(85)
    elif stage in ('fine', 'fine_half'):
        fine_worker(m, index, half=stage == 'fine_half')
    elif stage == 'review':
        review(m)
    else:
        raise ValueError('unknown stage')


def prepare():
    design = yaml.safe_load((ROOT/'conf/gate_f/prerequisites.yaml').read_text())
    run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    output = ROOT/'output/gate_f/prerequisites'/run_id
    configs = ROOT/'conf/frozen/gate_f_prerequisites'/run_id
    raw = ROOT/'data/research/gate_f_prerequisites'/run_id
    snapshot = ROOT/'scripts/archive'/('gate_f_prerequisites_'+run_id)/'code'
    for path in (output, configs, raw, snapshot):
        path.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in SOURCE_FILES:
        destination = snapshot/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, destination)
        hashes[name] = sha256(destination)
    material = snapshot/'conf/literature/gomonay_2024.yaml'
    # Bounded CPU timing of the actual streaming writer, including observer/I/O.
    timings = []
    for size, method in [(4, 'heun_projected_predictor'), (4, 'geometric_midpoint'),
                         (16, 'heun_projected_predictor'), (32, 'heun_projected_predictor')]:
        c = configuration(design, run_id+'_timing', material, size, 1.5, .64, .0025, .16,
                          8, method, design['seed']+900000+size)
        path = output/f'timing_{size}_{method}.yaml'
        with path.open('x') as h:
            yaml.safe_dump(c, h)
        started = time.monotonic()
        _, done = run_llg(path, output/f'timing_{size}_{method}.h5', device='cpu')
        if not done:
            raise RuntimeError('timing did not complete')
        timings.append(dict(size=size, method=method, seconds=time.monotonic()-started, steps=c['steps']))
    matrix = coupling_matrix(4, load_runtime(material).reduced)
    sample_mc(matrix, .047/11.1, 1., 2, 1, 1, 0)  # exclude Numba compile from throughput
    started = time.monotonic()
    sample_mc(matrix, .047/11.1, 1., 2000, 20, 2, 0)
    mc_seconds = time.monotonic()-started
    def minutes(seconds):
        return max(30, math.ceil((seconds*design['budget']['walltime_margin']+600)/60))
    kernel_time = max(t['seconds']/t['steps']*(design['kernel']['duration']/method['dt'])
                      for t in timings if t['size'] == 4 for method in design['kernel']['methods'] if method['method'] == t['method'])
    calibration_time = max(t['seconds']/t['steps']*design['calibration']['duration']/design['calibration']['dt'] for t in timings if t['size'] > 4)
    fine_time = max(t['seconds']/t['steps']*design['fine_window']['duration']/design['fine_window']['dt']*1.5 for t in timings if t['size'] > 4)
    requests = dict(mc=minutes(mc_seconds*design['kernel']['mc_sweeps']/2000*design['kernel']['chains']),
                    kernel=minutes(kernel_time), gate=30, calibration=minutes(calibration_time),
                    fine=minutes(fine_time), fine_half=minutes(2*fine_time), review=30)
    counts = dict(mc=6, kernel=24, gate=1, calibration=6, fine=6, fine_half=6, review=1)
    hours = sum(requests[k]*counts[k] for k in requests)/60
    # Conservative uncompressed spin/observable bound with 2x reserve for I/O/metadata.
    spin_bytes = (24*16001*8*32*3*8 + 6*40000*8*32*3*8
                  + 3*8001*8*(512+2048)*3*8 + 6*1601*12*(512+2048)*3*8)
    storage = spin_bytes*2/2**30
    resources = dict(timings=timings, mc_seconds_per_2000_sweeps=mc_seconds, requests_minutes=requests,
                     counts=counts, total_allocated_cpu_hours=hours, storage_bound_gib=storage,
                     assumption='short measured CPU timings extrapolated, 1.5 walltime margin; not completion guarantees')
    dump(output/'resource_estimate.json', resources)
    if hours > design['budget']['max_allocated_cpu_hours'] or max(requests.values()) > design['budget']['max_job_minutes']:
        raise RuntimeError('measured resource budget exceeded; no jobs submitted')
    if storage > design['budget']['max_storage_gib'] or shutil.disk_usage(raw).free < (storage+10)*2**30:
        raise RuntimeError('storage budget exceeded; no jobs submitted')
    config_hashes = {}
    for i, theta in enumerate(design['candidate_theta']):
        for j, method in enumerate(design['kernel']['methods']):
            index = i*4+j
            c = configuration(design, run_id+f'_kernel_{index}', material, 4, theta,
                              design['kernel']['duration'], method['dt'], design['kernel']['save_dt'],
                              design['kernel']['chains'], method['method'], design['seed']+1000+index*100)
            path = configs/f'kernel_{index}.yaml'
            with path.open('x') as h:
                yaml.safe_dump(c, h)
            config_hashes[path.name] = sha256(path)
    old_gate = ROOT/'conf/frozen/zero_field_gomonay/20260911_040025/equilibrium_gate.json'
    old = json.loads(old_gate.read_text())
    old_rows = []
    for index, obs in old['comparisons'].items():
        for name, values in obs.items():
            if 'simultaneous_ci' in values:
                old_rows.append(dict(index=index, observable=name, difference=values['difference'],
                    ci=values['simultaneous_ci'], llg_bulk_ess=values['llg']['bulk_ess'], mc_bulk_ess=values['mc']['bulk_ess']))
    dump(output/'historical_gate_diagnosis.json', dict(source_sha256=sha256(old_gate),
         status='descriptive_decomposition_not_new_certificate', original_status=old['status'], rows=old_rows,
         conclusion='small point differences coexist with broad CIs/low ESS; no proven implementation fault or new pass'))
    manifest = dict(run_id=run_id, design=design, output=str(output), raw=str(raw), snapshot=str(snapshot),
                    configs=str(configs), material=str(material), files_sha256=hashes,
                    configs_sha256=config_hashes, resources=resources, production_enabled=False,
                    paper_reference_sha256=sha256(ROOT/'data/literature_reproduction/gomonay_2024/reference/Gomonay_2024_supplement.pdf'))
    order_rng = np.random.default_rng(design['seed'])
    manifest['execution_order'] = {stage: order_rng.permutation(count).tolist() for stage, count in counts.items() if count > 1}
    path = configs/'manifest.json'
    dump(path, manifest)
    print(json.dumps(dict(manifest=str(path), resources=resources), indent=2))


def submit(manifest):
    manifest = Path(manifest).resolve()
    m = json.loads(manifest.read_text())
    submission = Path(m['output'])/'submission.json'
    if submission.exists():
        raise FileExistsError('already submitted or partial submission; inspect before retry')
    logs = ROOT/'logs/gate_f_prerequisites'/m['run_id']
    logs.mkdir(parents=True, exist_ok=False)
    jobs = []
    def record(status):
        temp = submission.with_suffix('.json.partial')
        temp.write_text(json.dumps(dict(status=status, manifest_sha256=sha256(manifest), jobs=jobs,
                         no_monitoring=True, production_enabled=False), indent=2))
        temp.replace(submission)
    record('submitting')
    for stage in ['mc', 'kernel', 'gate', 'calibration', 'fine', 'fine_half', 'review']:
        count = m['resources']['counts'][stage]
        command = ['sbatch', '--parsable', '--time='+str(m['resources']['requests_minutes'][stage]),
                   '--kill-on-invalid-dep=yes', '--output='+str(logs/(stage+'_%A_%a.log'))]
        if count > 1:
            command += [f'--array=0-{count-1}%2']
        if jobs:
            command += [('--dependency=afterany:' if stage == 'review' else '--dependency=afterok:')+jobs[-1]['job_id']]
        command += [str(Path(m['snapshot'])/'slurm/gate_f_prerequisites.sbatch'), m['snapshot'],
                    str(ROOT.parent/'run_zrs_mag.sh'), str(manifest), stage]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            record('partial_submission_failed_no_automatic_retry')
            raise RuntimeError(result.stderr)
        job_id = result.stdout.strip().split(';')[0]
        if not job_id.isdigit():
            raise RuntimeError('unexpected scheduler response; inspect before retry')
        jobs.append(dict(stage=stage, count=count, job_id=job_id, command=command))
        record('submitting')
    record('submitted_not_monitored')
    print(json.dumps(dict(submission=str(submission), jobs=[{k: j[k] for k in ('stage', 'count', 'job_id')} for j in jobs]), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest')
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--submit', action='store_true')
    p.add_argument('--stage', choices=['mc', 'kernel', 'gate', 'calibration', 'fine', 'fine_half', 'review'])
    p.add_argument('--index', type=int, default=0)
    args = p.parse_args()
    if args.prepare:
        prepare()
    elif args.submit:
        submit(args.manifest)
    else:
        worker(args.manifest, args.stage, args.index)


if __name__ == '__main__':
    main()
