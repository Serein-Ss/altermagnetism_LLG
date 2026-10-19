"""Bounded paper-evidence completion. Frozen production files remain untouched."""
import argparse
import copy
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial.distance import cdist
from scripts.workflow.bauer_campaign import digest, save
from scripts.model.sphere import sphere_exp
from tests.experiments.director_transition import DirectorTransition
from tests.experiments.projected_midpoint_sampler import sample

ROOT = Path('tests/paper_work/completion')
CONFIG = ROOT/'config.yaml'


def cfg():
    return yaml.safe_load(CONFIG.read_text())


def energy_distance(a, b):
    a = a.reshape(len(a), -1).astype(float)/np.sqrt(a.shape[-2])
    b = b.reshape(len(b), -1).astype(float)/np.sqrt(b.shape[-2])
    return float(2*cdist(a, b).mean()-cdist(a, a).mean()-cdist(b, b).mean())


def subset_indices(rows, fraction, seed):
    """Nested, source-stratified WHOLE-path subsets; retain every time window."""
    groups = {}
    for r in rows:
        if r['split'] == 'train':
            groups.setdefault(r['source'], {}).setdefault(r['path_index'], []).append(int(r['sample']))
    rng = np.random.default_rng(seed); chosen = []
    for source in sorted(groups):
        paths = sorted(groups[source], key=int)
        selected = rng.permutation(paths)[:round(len(paths)*fraction)]
        for path in selected:
            chosen.extend(groups[source][path])
    return np.array(sorted(chosen), dtype=int)


def prepare():
    c = cfg()
    if (ROOT/'manifest.json').exists():
        raise FileExistsError('Preserve frozen campaign manifest')
    for name in ('data', 'output', 'results', 'logs', 'figures', 'tables'):
        (ROOT/name).mkdir(parents=True, exist_ok=True)
    states = []; rows = []; rng = np.random.default_rng(c['state_seed'])
    for length in c['sites']:
        for theta in c['theta']:
            for kind in c['kinds']:
                x = np.zeros((length, 3))
                if kind == 'aligned':
                    x[:, 2] = 1
                elif kind == 'tilted':
                    x[:, 0] = np.sin(.8); x[:, 2] = np.cos(.8)
                else:
                    # Fixed physical wall width, not stretched with system size.
                    z = (np.arange(length)-(length-1)*.5)/np.sqrt(1/c['anisotropy'])
                    x[:, 2] = np.tanh(z); x[:, 0] = 1/np.cosh(z)
                x += .025*rng.standard_normal(x.shape)
                x /= np.linalg.norm(x, axis=-1, keepdims=True)
                i = len(rows); path = ROOT/'data'/f'state{i}.npz'
                np.savez_compressed(path, spins=x)
                states.append(x)
                rows.append(dict(state=i, sites=length, theta=theta, kind=kind,
                                 path=str(path), sha256=digest(path)))
    with open('tests/paper_work/tables/T01_transition_index.csv') as f:
        index = list(csv.DictReader(f))
    subsets = {str(f): subset_indices(index, f, c['subset_seed']).tolist() for f in c['fractions']}
    sources = {r['source'] for r in index}
    sources.update([c['frozen_checkpoint'], str(CONFIG), __file__,
        str(ROOT/'analyze.py'), str(ROOT/'long_rollout.py'),
        'tests/experiments/director_transition.py', 'tests/experiments/local_equivariant_transition.py',
        'tests/experiments/projected_midpoint_sampler.py', 'scripts/model/sphere.py',
        'scripts/model/gate_f.py', 'scripts/core/bauer_fast.py',
        'scripts/literature/bauer_2011/integrator.py', 'scripts/core/reduced_llg.py',
        'tests/experiments/independent_reference.py',
        'tests/experiments/coverage_transition_dataset.py',
        'tests/experiments/fine_transition_dataset.py',
        'tests/conf/local_transition_training.json',
        'tests/experiments/reflection_augmentation.py',
        'tests/experiments/director_budget.py',
        'tests/experiments/sample_tangent_baseline.py',
        'tests/experiments/sample_tangent_path_comparison.py',
        'tests/output/sample_tangent_baseline/model.json'])
    tasks = []
    for method in ('mt', 'heun'):
        for r in rows:
            tasks.append(dict(arm=method, state=r['state'], duration=c['duration']))
        for state in c['long_states']:
            tasks.append(dict(arm=method, state=state, duration=c['long_duration']))
    for state in range(6):
        tasks.append(dict(arm='mt_fine', state=state, duration=8.))
    for state in (0, 3, 6, 9, 12, 15):
        tasks.append(dict(arm='calibration', state=state, duration=c['long_duration']))
    training = [dict(kind='flow', seed=s, fraction=f) for f in c['fractions'] for s in c['seeds']]
    training += [dict(kind='deterministic', seed=s, fraction=1.) for s in c['seeds']]
    generated = [dict(arm=a, state=r['state'], duration=c['duration'])
                 for a in ('frozen', 'selected', 'deterministic') for r in rows]
    generated += [dict(arm=a, state=s, duration=c['long_duration'])
                  for a in ('frozen', 'selected', 'deterministic') for s in c['long_states']]
    save(ROOT/'manifest.json', dict(config=c, states=rows, subsets=subsets,
         training=training, reference=tasks, generation=generated,
         sources={p:digest(p) for p in sorted(sources)},
         scope='Exploratory: 64 independent noise paths per fixed initial; not 64 independent initial states. '
         'Test states are fresh analytic perturbations, not equilibrated samples. No test-driven selection. '
         'Development has historical training-container lineage; not final external certification. '
         'Three seeds at each fraction; constant updates, hence different reuse counts.'))


def load():
    m = json.loads((ROOT/'manifest.json').read_text())
    for p, h in m['sources'].items():
        if digest(p) != h:
            raise ValueError(f'Frozen source changed: {p}')
    return m


def initial(m, state):
    r = m['states'][state]
    assert digest(r['path']) == r['sha256']
    with np.load(r['path']) as z:
        return z['spins'].copy()


def require_gpu():
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Slurm GPU allocation required')
    if os.getenv('SLURM_JOB_PARTITION') != 'rtx4090':
        raise RuntimeError('rtx4090 partition required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def model_load(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = DirectorTransition(ck['theta_mean'], ck['theta_std'], ck['config']['width']).to(device)
    model.load_state_dict(ck['ema']); model.eval()
    return model, ck


def deterministic_step(model, x, temperature, lag, axis):
    return sphere_exp(x, model(x, x.new_ones(len(x)), x, temperature, lag, axis))


@torch.no_grad()
def paths(model, ck, x0, theta, duration, n, seed, kind='flow', steps=32, device='cuda', save_dt=.01):
    x = torch.as_tensor(x0, dtype=torch.float32, device=device).expand(n, -1, -1).clone()
    x = torch.nn.functional.normalize(x, dim=-1)
    lag = ck['config']['lag']; stride = round(save_dt/lag)
    assert stride >= 1 and np.isclose(stride*lag, save_dt)
    count = round(duration/lag)
    assert count % stride == 0 and np.isclose(count*lag, duration)
    t = x.new_full((n,), theta); dt = x.new_full((n,), lag)
    axis = x.new_tensor([0., 0., 1.]).expand(n, -1)
    rng = torch.Generator(device=device).manual_seed(seed)
    frames = [x.cpu().numpy()]
    for i in range(count):
        x = deterministic_step(model, x, t, dt, axis) if kind == 'deterministic' else sample(
            model, x, t, dt, axis, rng, ck['latent_scale'], steps)
        if (i+1) % stride == 0:
            frames.append(x.cpu().numpy())
    return np.stack(frames, axis=1)


def persist(name, p, dt, meta, seconds):
    if not np.isfinite(p).all() or np.max(abs(np.linalg.norm(p, axis=-1)-1)) >= 1e-5:
        raise ValueError('Numerical failure; not a censored observation')
    path = ROOT/'data'/f'{name}.npz'
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, spins=p, reduced_time=np.arange(p.shape[1])*dt)
    save(ROOT/'results'/f'{name}.json', dict(status='COMPLETE_NUMERICAL_ONLY', path=str(path),
         sha256=digest(path), metadata=meta, seconds_excluding_save=seconds,
         manifest_sha256=digest(ROOT/'manifest.json'), job_id=os.getenv('SLURM_JOB_ID')))


def reference(index):
    from tests.experiments.independent_reference import generate
    m = load(); c = m['config']; task = m['reference'][index]; r = m['states'][task['state']]
    x = np.repeat(initial(m, task['state'])[None], c['paths'], axis=0)
    seeds = c['reference_seed_base']+index*10000+np.arange(len(x))
    dt = c['reference_dt']/(2 if task['arm'] == 'mt_fine' else 1)
    save_dt = c['save_dt']
    method = task['arm'].split('_')[0]
    if task['arm'] == 'calibration':
        x[:] = 0; x[:len(x)//2,:,2] = 1; x[len(x)//2:,:,2] = -1
        method = 'mt'; save_dt = .1
    started = time.monotonic()
    p, meta = generate(x, seeds, dt=dt, duration=task['duration'], save_dt=save_dt,
                      anisotropy=c['anisotropy'], damping=c['damping'], theta=r['theta'],
                      method=method)
    persist(f'reference_{index}', p, save_dt, dict(task=task, numerics=meta), time.monotonic()-started)


def baseline(index):
    from tests.experiments.sample_tangent_path_comparison import rollout
    m = load(); c = m['config']
    tasks = [t for t in m['generation'] if t['arm'] == 'frozen']
    task = dict(tasks[index], arm='gaussian'); state = task['state']
    model = json.loads(Path('tests/output/sample_tangent_baseline/model.json').read_text())
    theta = m['states'][state]['theta']
    param = next(r for r in model['rows'] if r['theta'] == theta)
    started = time.monotonic()
    p = rollout(initial(m, state), param, np.asarray(model['axis']), c['paths'],
                round(task['duration']/c['lag']), 3500000000+state)
    persist(f'baseline_{index}', p, c['save_dt'], dict(task=task), time.monotonic()-started)


def validate(model, ck, kind):
    from tests.experiments.fine_transition_dataset import FineTransitionDataset
    rows = []
    for i, r in enumerate(FineTransitionDataset('development').rows):
        assert digest(r['path']) == r['sha256']
        with np.load(r['path']) as z:
            real = z['spins']
        p = paths(model, ck, real[0, 0], r['state']['theta'], .08, 64,
                  3300000000+i, kind=kind)
        rows.append(dict(anchor=r['anchor'], score=energy_distance(p[:, -1], real[:, -1])))
    return float(np.mean([r['score'] for r in rows])), rows


def train(index):
    from tests.experiments.coverage_transition_dataset import CoverageTransitionDataset
    from tests.experiments.director_budget import training_arrays
    from tests.experiments.reflection_augmentation import augment_pairs
    from tests.experiments.local_equivariant_transition import loss
    require_gpu(); m = load(); c = m['config']; spec = m['training'][index]
    out = ROOT/'output'/f'train{index}'; out.mkdir(exist_ok=False)
    arrays = training_arrays(CoverageTransitionDataset())
    ids = m['subsets'][str(spec['fraction'])]
    a, b, theta = [v[ids] for v in arrays[:3]]
    tangent = b-a-((b-a)*a).sum(-1, keepdims=True)*a
    scale = float(np.sqrt(np.mean(np.sum(tangent*tangent, axis=-1))/2))
    mean, std = float(theta.mean()), float(theta.std())
    tc = json.loads(Path('tests/conf/local_transition_training.json').read_text())
    tc.update(updates=c['updates'], flow_steps=c['flow_steps'])
    torch.manual_seed(spec['seed'])
    model = DirectorTransition(mean, std, tc['width']).cuda(); ema = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, c['updates'], eta_min=tc['final_learning_rate'])
    a, b, theta = [torch.as_tensor(v, dtype=torch.float32, device='cuda') for v in (a, b, theta)]
    a = torch.nn.functional.normalize(a, dim=-1); b = torch.nn.functional.normalize(b, dim=-1)
    rng = torch.Generator(device='cuda').manual_seed(3400000000+spec['seed'])
    aug = torch.Generator(device='cuda').manual_seed(3410000000+spec['seed'])
    axis = a.new_tensor(tc['axis']).expand(tc['batch_size'], -1)
    lag = a.new_full((tc['batch_size'],), c['lag'])
    report = dict(spec=spec, pair_count=len(a), path_count=len(a)//8, history=[], validation=[],
                  status='TRAINING', selection=c['selection'], selected_score=None)
    best = float('inf'); started = time.monotonic()
    for step in range(1, c['updates']+1):
        ix = torch.randint(len(a), (tc['batch_size'],), generator=rng, device='cuda')
        x, y, _ = augment_pairs(a[ix], b[ix], aug)
        optimizer.zero_grad(set_to_none=True)
        if spec['kind'] == 'flow':
            value, _ = loss(model, x, y, theta[ix], lag, axis, rng, scale)
        else:
            value = (deterministic_step(model, x, theta[ix], lag, axis)-y).square().sum(-1).mean()
        value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        with torch.no_grad():
            for ep, p in zip(ema.parameters(), model.parameters()):
                ep.lerp_(p, 1-tc['ema_decay'])
        if step % 100 == 0:
            report['history'].append(dict(step=step, loss=float(value.detach())))
        if step % c['validation_every'] == 0:
            ck = dict(ema=ema.state_dict(), theta_mean=mean, theta_std=std, latent_scale=scale,
                      config=tc, step=step, kind=spec['kind'])
            score, rows = validate(ema.eval(), ck, spec['kind'])
            report['validation'].append(dict(step=step, score=score, anchors=rows))
            torch.save(ck, out/f'step{step}.pt')
            if score < best:
                best = score; torch.save(ck, out/'selected.pt')
                report.update(selected_score=score, selected_step=step)
            save(ROOT/'results'/f'train{index}.json', report)
    torch.cuda.synchronize()
    report.update(status='COMPLETE_TRAINING_NOT_CERTIFIED', seconds_training_and_validation=time.monotonic()-started,
                  checkpoint=str(out/'selected.pt'), sha256=digest(out/'selected.pt'),
                  manifest_sha256=digest(ROOT/'manifest.json'))
    save(ROOT/'results'/f'train{index}.json', report)
    # All candidate short-time tests retained, not used in checkpoint or display selection.
    selected, ck = model_load(out/'selected.pt', 'cuda')
    for state in range(6):
        started = time.monotonic()
        p = paths(selected, ck, initial(m, state), m['states'][state]['theta'], .08,
                  c['paths'], c['latent_seed_base']+state, kind=spec['kind'])
        persist(f'train{index}_test{state}', p, c['save_dt'], dict(training=index, state=state), time.monotonic()-started)


def selected_checkpoint(m, kind):
    # Choose seed and checkpoint only by fixed development criterion, never new test results.
    candidates = []
    for i, spec in enumerate(m['training']):
        if spec['kind'] == kind and spec['fraction'] == 1.:
            r = json.loads((ROOT/'results'/f'train{i}.json').read_text())
            assert r['status'] == 'COMPLETE_TRAINING_NOT_CERTIFIED'
            assert digest(r['checkpoint']) == r['sha256']
            candidates.append((r['selected_score'], i, r['checkpoint']))
    return min(candidates)[2]


def generate_task(index):
    require_gpu(); m = load(); c = m['config']; task = m['generation'][index]
    arm = task['arm']; kind = 'deterministic' if arm == 'deterministic' else 'flow'
    path = c['frozen_checkpoint'] if arm == 'frozen' else selected_checkpoint(m, kind)
    model, ck = model_load(path, 'cuda'); state = task['state']
    torch.cuda.synchronize(); started = time.monotonic()
    p = paths(model, ck, initial(m, state), m['states'][state]['theta'], task['duration'],
              c['paths'], c['latent_seed_base']+state, kind=kind, steps=c['flow_steps'])
    torch.cuda.synchronize()
    persist(f'generation_{index}', p, c['save_dt'], dict(task=task, checkpoint=path,
            checkpoint_sha256=digest(path)), time.monotonic()-started)


def benchmark():
    from scripts.core.reduced_llg import BondHamiltonian
    from scripts.literature.bauer_2011.integrator import BauerWeakRK
    require_gpu(); m = load(); c = m['config']; model, ck = model_load(c['frozen_checkpoint'], 'cuda')
    rows = []
    for length in c['sites']:
        bonds = [(i, i+1) for i in range(length-1)]
        solver = BauerWeakRK(BondHamiltonian(bonds, 1., anisotropy=.1), alpha=.1, theta=.11)
        state = next(r['state'] for r in m['states'] if r['sites'] == length)
        for n in (1, 64, 128):
            x0 = initial(m, state)
            for arm in ('flow', 'mt'):
                for repeat in range(4):
                    x = torch.tensor(x0, device='cuda', dtype=torch.float64 if arm == 'mt' else torch.float32).expand(n,-1,-1).clone()
                    rng = torch.Generator(device='cuda').manual_seed(3420000000+repeat)
                    t = x.new_full((n,), .11); lag = x.new_full((n,), .01)
                    axis = x.new_tensor([0.,0.,1.]).expand(n,-1)
                    torch.cuda.synchronize(); started = time.monotonic()
                    with torch.no_grad():
                        for _ in range(8 if arm == 'flow' else 64):
                            if arm == 'flow':
                                x = sample(model, x, t, lag, axis, rng, ck['latent_scale'], 32)
                            else:
                                x, _ = solver.step(x, .00125, solver.noise_increment(x, .00125, rng))
                    torch.cuda.synchronize(); seconds = time.monotonic()-started
                    assert torch.isfinite(x).all() and (x.norm(dim=-1)-1).abs().max() < 1e-5
                    if repeat:
                        rows.append(dict(arm=arm, sites=length, paths=n, repeat=repeat, duration=.08,
                                         seconds=seconds, dtype=str(x.dtype), device=torch.cuda.get_device_name()))
    save(ROOT/'results/benchmark.json', dict(rows=rows, status='COMPLETE_TIMING_NOT_EQUAL_ACCURACY',
         scope='Same allocation, no I/O, first repeat warmup. MT float64, model float32; accuracy separately evaluated. '
         'Linear long-time projections are estimates, not observed runtimes.',
         literature_duration=c['literature_duration'], physical_transitions=round(c['literature_duration']/c['lag'])))


def smoke():
    require_gpu(); m = load(); model, ck = model_load(m['config']['frozen_checkpoint'], 'cuda')
    for state in (0, 6, 12):
        p = paths(model, ck, initial(m,state), .11, .02, 4, 123)
        assert np.isfinite(p).all() and np.max(abs(np.linalg.norm(p,axis=-1)-1)) < 1e-5
    from tests.experiments.local_equivariant_transition import loss
    for kind in ('flow', 'deterministic'):
        trial=DirectorTransition(.12,.01,64).cuda()
        x=torch.tensor(initial(m,0),device='cuda',dtype=torch.float32).expand(4,-1,-1).clone()
        x=torch.nn.functional.normalize(x,dim=-1)
        y=torch.roll(x,1,1);theta=x.new_full((4,),.11);lag=x.new_full((4,),.01)
        axis=x.new_tensor([0.,0.,1.]).expand(4,-1)
        if kind=='flow':
            value,_=loss(trial,x,y,theta,lag,axis,torch.Generator(device='cuda').manual_seed(44),.015)
        else:
            value=(deterministic_step(trial,x,theta,lag,axis)-y).square().mean()
        value.backward()
        assert torch.isfinite(value) and all(torch.isfinite(p.grad).all() for p in trial.parameters() if p.grad is not None)
    save(ROOT/'results/smoke.json', dict(status='PASS', manifest_sha256=digest(ROOT/'manifest.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare','reference','baseline','train','generate','benchmark','smoke'])
    p.add_argument('--index', type=int, default=0); p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    a = p.parse_args()
    if a.stage not in ('prepare', 'reference', 'baseline') and a.device != 'cuda':
        raise ValueError('--device cuda required')
    if a.stage != 'prepare' and not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('Submit computation through Slurm')
    if a.stage in ('reference','baseline','train','generate'):
        {'reference':reference,'baseline':baseline,'train':train,'generate':generate_task}[a.stage](a.index)
    else:
        {'prepare':prepare,'benchmark':benchmark,'smoke':smoke}[a.stage]()
