"""Fresh initial-state bank and longer independent-noise numerical references."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.independent_reference import generate

CONFIG = Path('tests/conf/fresh_rollout_reference.json')
ROOT = Path('tests/results/fresh_rollout_reference')
RAW = Path('tests/data/fresh_rollout_reference')


def initial_bank(cfg):
    rng = np.random.default_rng(cfg['initial_seed'])
    states, rows = [], []
    for kind in cfg['kinds']:
        for family in range(cfg['families_per_kind']):
            n = cfg['sites']
            base = np.zeros((n, 3))
            if kind == 'aligned':
                base[:, 2] = 1
            elif kind == 'tilted':
                base[:, 0] = np.sin(cfg['tilt_radians'])
                base[:, 2] = np.cos(cfg['tilt_radians'])
            elif kind == 'wall':
                z = (np.arange(n)-(n-1)/2)/cfg['wall_width_sites']
                base[:, 0] = 1/np.cosh(z)
                base[:, 2] = np.tanh(z)
            else:
                raise ValueError('unknown initial kind')
            sigma = cfg['aligned_noise'] if kind == 'aligned' else cfg['texture_noise']
            s = base+sigma*rng.normal(size=base.shape)
            s /= np.linalg.norm(s, axis=-1, keepdims=True)
            states.append(s)
            rows.append(dict(kind=kind, family=family, initial_id=len(states)-1))
    return np.asarray(states), rows


def prepare():
    cfg = json.loads(CONFIG.read_text())
    if ROOT.exists() or RAW.exists():
        raise FileExistsError('preserve prior validation')
    frozen = []
    for seed in [11, 22, 33]:
        p = Path(f'tests/results/director_budget/24000/{seed}/summary.json')
        r = json.loads(p.read_text())
        assert r['status'] == 'COMPLETE_TRAINING_AND_GENERATION_NOT_CERTIFIED'
        assert digest(r['checkpoint']) == r['checkpoint_sha256']
        frozen.append(dict(seed=seed, checkpoint=r['checkpoint'], sha256=r['checkpoint_sha256'],
                           report=str(p), report_sha256=digest(p)))
    states, families = initial_bank(cfg)
    with np.load('tests/data/fine_transition_pilot/anchors.npz') as z:
        old = z['unit_initials']
    assert all(not np.array_equal(a, b) for a in states for b in old)
    anchors = [dict(row, theta=theta) for theta in cfg['theta'] for row in families]
    schedule = [(c, a) for c in range(len(cfg['cases'])) for a in range(len(anchors))]
    order = np.random.default_rng(cfg['schedule_seed']).permutation(len(schedule))
    ROOT.mkdir(); RAW.mkdir()
    bank = RAW/'anchors.npz'
    np.savez_compressed(bank, unit_initials=states)
    save(ROOT/'manifest.json', dict(config=cfg, config_sha256=digest(CONFIG),
        anchors=anchors, bank_sha256=digest(bank), frozen_models=frozen,
        schedule=[schedule[i] for i in order], total_paths=len(schedule)*cfg['paths_per_anchor'],
        code_sha256={p: digest(p) for p in [__file__, 'tests/experiments/independent_reference.py', 'scripts/core/bauer_fast.py']}))


def run():
    if not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('CPU Slurm required')
    manifest = json.loads((ROOT/'manifest.json').read_text())
    cfg = manifest['config']
    assert digest(CONFIG) == manifest['config_sha256']
    assert digest(RAW/'anchors.npz') == manifest['bank_sha256']
    for p, h in manifest['code_sha256'].items():
        assert digest(p) == h
    dest = ROOT/'summary.json'
    if dest.exists():
        raise FileExistsError('no silent retry')
    with np.load(RAW/'anchors.npz') as z:
        states = z['unit_initials']
    report = dict(status='RUNNING', rows=[], manifest_sha256=digest(ROOT/'manifest.json'), job_id=os.environ['SLURM_JOB_ID'])
    save(dest, report)
    for ci, ai in manifest['schedule']:
        a = manifest['anchors'][ai]
        n = cfg['paths_per_anchor']
        seeds = cfg['noise_base']+(ci*len(manifest['anchors'])+ai)*10000+np.arange(n)
        initial = np.repeat(states[a['initial_id']:a['initial_id']+1], n, axis=0)
        paths, numerics = generate(initial, seeds, **cfg['cases'][ci], duration=cfg['duration'],
            save_dt=cfg['save_dt'], anisotropy=cfg['anisotropy'], damping=cfg['damping'], theta=a['theta'])
        p = RAW/f'case{ci}_anchor{ai}.npz'
        np.savez_compressed(p, spins=paths, seeds=seeds, reduced_time=np.arange(paths.shape[1])*cfg['save_dt'])
        report['rows'].append(dict(case=ci, anchor=ai, path=str(p), sha256=digest(p), numerics=numerics))
        save(dest, report)
    report['status'] = 'COMPLETE_FRESH_ROLLOUT_REFERENCE_NOT_CERTIFIED'
    save(dest, report)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepare', action='store_true')
    args = p.parse_args()
    prepare() if args.prepare else run()
