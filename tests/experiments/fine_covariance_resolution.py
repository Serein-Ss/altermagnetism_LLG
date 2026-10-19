"""Matched-count covariance diagnostics; repeated partitions are not new data."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.fine_increment_statistics import increments

ROOT = Path('tests/results/fine_covariance_resolution')
LAGS = np.array([.01, .02, .04, .08])
SEED = 940000000
PARTITIONS = 100
METRICS = ['all_entries', 'same_site_blocks', 'different_site_blocks', 'nearest_neighbor_blocks']


def covariance(delta):
    n, k, sites, xyz = delta.shape
    if n < 2 or xyz != 3 or k != len(LAGS) or not np.isfinite(delta).all():
        raise ValueError('invalid increments')
    flat = delta.reshape(n, k, sites * 3).transpose(1, 0, 2)
    centered = flat - flat.mean(axis=1, keepdims=True)
    return (centered.transpose(0, 2, 1) @ centered) / (n-1) / LAGS[:, None, None]


def errors(a, b):
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] % 3:
        raise ValueError('incompatible covariance matrices')
    sites = np.arange(a.shape[-1]) // 3
    distance = np.abs(sites[:, None] - sites[None, :])
    masks = [np.ones_like(distance, dtype=bool), distance == 0, distance > 0, distance == 1]
    return np.stack([np.sqrt(np.mean((a-b)[:, mask]**2, axis=1)) for mask in masks], axis=-1)


def split_covariances(delta, rng):
    if len(delta) % 2:
        raise ValueError('even path count required')
    indices = rng.permutation(len(delta))
    a, b = np.split(indices, 2)
    return covariance(delta[a]), covariance(delta[b])


def run():
    source = Path('tests/results/fine_transition_pilot/summary.json')
    report = json.loads(source.read_text())
    assert report['status'] == 'COMPLETE_FINE_TRANSITION_PILOT_NOT_CERTIFIED'
    rows = {(r['case'], r['anchor']): r for r in report['rows']}
    assert len(report['rows']) == len(rows) == 48
    assert set(rows) == {(c, a) for c in range(4) for a in range(12)}
    ROOT.mkdir(exist_ok=False)
    result = dict(status='RUNNING', source_sha256=digest(source), code_sha256=digest(__file__),
        seed=SEED, partitions=PARTITIONS, paths_per_half=128, lags=LAGS.tolist(), metrics=METRICS,
        scope='All comparisons use 128 independent noise paths per covariance estimate. '
        'Random partitions reuse the same 256 paths: their quantiles are partition sensitivity, NOT confidence intervals, p-values or certification. '
        'Anchor-conditional only; no pooled population inference. Finite-lag tangent chord covariance, not full transition-law validation.', rows=[])
    for anchor in range(12):
        deltas = []
        for case in range(4):
            row = rows[case, anchor]
            assert digest(row['path']) == row['sha256']
            with np.load(row['path']) as d:
                assert len(d['spins']) == 256
                if case == 0:
                    initial = d['spins'][0, 0].copy()
                _, delta = increments(d['spins'], d['reduced_time'], initial, LAGS)
            deltas.append(delta)
        rng = np.random.default_rng(SEED+anchor)
        within = []; between = []
        for _ in range(PARTITIONS):
            pairs = [split_covariances(delta, rng) for delta in deltas]
            within.append([errors(a, b) for a, b in pairs])
            between.append([errors(pairs[c][0], pairs[2][0]) for c in [0, 1, 3]])
        within, between = np.asarray(within), np.asarray(between)
        assert np.isfinite(within).all() and np.isfinite(between).all()
        path = ROOT / f'anchor{anchor}.npz'
        np.savez_compressed(path, within=within, between=between, lags=LAGS,
                            comparison_cases=np.array([0, 1, 3]), reference_case=np.array(2))
        result['rows'].append(dict(anchor=anchor, path=str(path), sha256=digest(path),
            source_sha256={str(c): rows[c, anchor]['sha256'] for c in range(4)},
            within_partition_quantiles=np.quantile(within, [.025, .5, .975], axis=0).tolist(),
            between_partition_quantiles=np.quantile(between, [.025, .5, .975], axis=0).tolist()))
        save(ROOT/'summary.json', result)
    result['status'] = 'COMPLETE_COVARIANCE_RESOLUTION_NOT_CERTIFIED'
    save(ROOT/'summary.json', result)


if __name__ == '__main__':
    run()
