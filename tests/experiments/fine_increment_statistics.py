"""Exploratory finite-lag moments, conditional on each fixed pilot anchor."""
import json
import os
from pathlib import Path

import numpy as np
from scripts.workflow.bauer_campaign import digest, save

SOURCE = Path('tests/results/fine_transition_pilot')
ROOT = Path('tests/results/fine_increment_statistics')
LAGS = np.array([.01, .02, .04, .08])
REPLICATES = 2000
SEED = 930000000


def increments(spins, times, anchor, lags):
    if spins.ndim != 4 or spins.shape[2:] != anchor.shape or spins.shape[1] != len(times):
        raise ValueError('incompatible shapes')
    if not np.isfinite(spins).all() or not np.isfinite(times).all():
        raise ValueError('nonfinite input')
    if times[0] != 0 or np.any(np.diff(times) <= 0) or np.any(np.asarray(lags) <= 0):
        raise ValueError('invalid time grid')
    if not np.array_equal(spins[:, 0], np.broadcast_to(anchor, spins[:, 0].shape)):
        raise ValueError('not a shared fixed initial state')
    if not np.allclose(np.linalg.norm(spins, axis=-1), 1, atol=1e-10, rtol=0):
        raise ValueError('nonunit spins')
    indices = []
    for lag in lags:
        match = np.flatnonzero(np.isclose(times, lag, atol=1e-12, rtol=0))
        if len(match) != 1:
            raise ValueError('lag must have exactly one saved frame; no interpolation')
        indices.append(match[0])
    ambient = spins[:, indices] - anchor
    tangent = ambient - np.sum(ambient * anchor, axis=-1, keepdims=True) * anchor
    return ambient, tangent


def moments(delta, lags):
    """Sample covariance / lag, not half covariance or an instantaneous generator."""
    n, k, sites, xyz = delta.shape
    if n < 2 or xyz != 3 or k != len(lags) or not np.isfinite(delta).all():
        raise ValueError('invalid increments')
    mean = delta.mean(axis=0)
    centered = (delta - mean).reshape(n, k, sites * 3)
    covariance = np.einsum('nki,nkj->kij', centered, centered) / (n - 1)
    return mean / lags[:, None, None], covariance / lags[:, None, None]


def bootstrap(delta, lags, weights):
    """Weights count whole paths; reuse across all sites, lags and projections."""
    n, k, sites, _ = delta.shape
    if weights.shape[1] != n or np.any(weights < 0) or not np.all(weights.sum(axis=1) == n):
        raise ValueError('invalid path bootstrap weights')
    mean = (weights @ delta.reshape(n, -1) / n).reshape(-1, k, sites, 3)
    second = (weights @ np.sum(delta**2, axis=-1).reshape(n, -1) / n).reshape(-1, k, sites)
    trace = (second - np.sum(mean**2, axis=-1)) * n / (n - 1) / lags[:, None]
    return mean / lags[:, None, None], trace


def run():
    report_path = SOURCE / 'summary.json'
    report = json.loads(report_path.read_text())
    manifest_path = SOURCE / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if report['status'] != 'COMPLETE_FINE_TRANSITION_PILOT_NOT_CERTIFIED' or digest(manifest_path) != report['manifest_sha256']:
        raise ValueError('incomplete or changed source')
    expected = {(c, a) for c in range(4) for a in range(12)}
    if len(report['rows']) != 48 or {(r['case'], r['anchor']) for r in report['rows']} != expected:
        raise ValueError('incomplete case coverage')
    bank = Path('tests/data/fine_transition_pilot/anchors.npz')
    if digest(bank) != manifest['bank_sha256']:
        raise ValueError('changed anchors')
    with np.load(bank) as data:
        anchors = data['unit_initials']
    ROOT.mkdir(exist_ok=False)
    result = dict(status='RUNNING', rows=[], source_sha256=digest(report_path),
                  code_sha256=digest(__file__), numpy_version=np.__version__,
                  lags=LAGS.tolist(), bootstrap_replicates=REPLICATES, bootstrap_seed=SEED,
                  job_id=os.getenv('SLURM_JOB_ID'),
                  scope='Finite-lag conditional moments, not instantaneous coefficients or model certification. '
                  'Pointwise percentile intervals, not simultaneous. Fixed anchors; only four parent families. '
                  'No pooling anchors or sites as independent replicates. Independent noise between numerical cases. '
                  'Tangent-projected chord increments, not logarithmic-map increments. Covariance rate has no factor 1/2.')
    for row in sorted(report['rows'], key=lambda r: (r['case'], r['anchor'])):
        ci, ai = row['case'], row['anchor']
        if digest(row['path']) != row['sha256']:
            raise ValueError('changed trajectories')
        with np.load(row['path']) as data:
            spins, times, seeds = data['spins'], data['reduced_time'], data['seeds']
        expected_seeds = manifest['config']['noise_base'] + (ci * 12 + ai) * 10000 + np.arange(256)
        if spins.shape[0] != 256 or not np.array_equal(seeds, expected_seeds):
            raise ValueError('unexpected paths or noise identities')
        ambient, tangent = increments(spins, times, anchors[ai], LAGS)
        seed = SEED + ci * 12 + ai
        weights = np.random.default_rng(seed).multinomial(256, np.full(256, 1 / 256), size=REPLICATES)
        arrays = dict(lags=LAGS, noise_seeds=seeds)
        for label, delta in [('ambient', ambient), ('tangent', tangent)]:
            mean, covariance = moments(delta, LAGS)
            bmean, btrace = bootstrap(delta, LAGS, weights)
            arrays.update({label+'_mean_rate': mean, label+'_covariance_rate': covariance,
                           label+'_mean_rate_ci95': np.quantile(bmean, [.025, .975], axis=0),
                           label+'_trace_rate_ci95': np.quantile(btrace, [.025, .975], axis=0),
                           label+'_site_average_trace_draws': btrace.mean(axis=-1)})
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError('nonfinite statistics')
        path = ROOT / f'case{ci}_anchor{ai}.npz'
        np.savez_compressed(path, **arrays)
        result['rows'].append(dict(case=ci, anchor=ai, state=manifest['anchors'][ai],
                                   source_sha256=row['sha256'], bootstrap_seed=seed,
                                   path=str(path), sha256=digest(path)))
        save(ROOT / 'summary.json', result)
    result['status'] = 'COMPLETE_FINITE_LAG_MOMENTS_NOT_CERTIFIED'
    save(ROOT / 'summary.json', result)


if __name__ == '__main__':
    run()
