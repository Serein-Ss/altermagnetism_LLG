"""Exploratory whole-path uncertainty and matched-count covariance diagnostics."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.director_uncertainty import weighted_profiles, differences
from tests.experiments.director_convergence_summary import full_covariance


def covariance_errors(a, b):
    dimension = a.shape[-1]
    sites = np.arange(dimension)//3
    masks = [np.ones((dimension, dimension), dtype=bool),
             sites[:, None] == sites[None, :],
             sites[:, None] != sites[None, :],
             abs(sites[:, None]-sites[None, :]) == 1]
    return np.stack([np.sqrt(np.mean((a-b)[:, mask]**2, axis=1))
                     for mask in masks], axis=-1)


def matched_covariance(generated, reference, rng, repeats=20):
    if generated.shape != reference.shape or len(reference) % 2:
        raise ValueError('equal even path counts required')
    n = len(reference)//2
    within, between = [], []
    for _ in range(repeats):
        ri = rng.permutation(2*n)
        gi = rng.permutation(2*n)[:n]
        ra = full_covariance(reference[ri[:n]])
        rb = full_covariance(reference[ri[n:]])
        gc = full_covariance(generated[gi])
        within.append(covariance_errors(ra, rb))
        between.append(covariance_errors(gc, ra))
    return np.asarray(within), np.asarray(between)


def run():
    reports = {}
    for seed in [11, 22, 33]:
        source = Path(f'tests/results/director_midpoint_sampling/{seed}/summary.json')
        report = json.loads(source.read_text())
        assert report['status'] == 'COMPLETE_MIDPOINT_REFINEMENT_NOT_CERTIFIED'
        selected = [r for r in report['rows'] if r['steps'] == 32]
        assert len(selected) == 12 and {r['anchor'] for r in selected} == set(range(12))
        reports[seed] = selected, digest(source)
    root = Path('tests/results/midpoint_distribution_audit')
    root.mkdir(exist_ok=False)
    result = dict(status='RUNNING', rows=[], replicates=2000, partition_repeats=20,
        covariance_metrics=['all_entries', 'same_site', 'different_site', 'nearest_neighbor'],
        code_sha256={p: digest(p) for p in [__file__,
            'tests/experiments/director_uncertainty.py',
            'tests/experiments/director_convergence_summary.py']},
        scope='Exploratory midpoint32 audit, fixed checkpoints/anchors. Independent whole-path '
        'bootstrap for generated and reference ensembles; pointwise percentile intervals, not '
        'simultaneous or equivalence tests. RMSE intervals are not calibrated zero-error tests. '
        'Train references reused in fitting. Development reused in model design. Repeated '
        '128-vs-128 partitions are sensitivity checks, not independent experiments or CIs. '
        'Adjacent-time connected covariance at first output time is structurally zero.')
    for seed, (rows, report_hash) in reports.items():
        for row in rows:
            data = []
            for key, hash_key, stride in [('path', 'sha256', 1),
                    ('reference_path', 'reference_sha256', 4)]:
                assert digest(row[key]) == row[hash_key]
                with np.load(row[key]) as d:
                    p = d['spins'][:, ::stride].astype(float)
                    t = d['reduced_time'][::stride]
                assert p.shape == (256, 9, 25, 3) and np.isfinite(p).all()
                assert np.max(abs(np.linalg.norm(p, axis=-1)-1)) < 1e-5
                if data:
                    np.testing.assert_allclose(t, times, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(p[:, 0], data[0][:, 0], atol=2e-7, rtol=0)
                data.append(p)
                times = t
            rng_seed = 990000000+seed*1000+row['anchor']
            rng = np.random.default_rng(rng_seed)
            boot = [weighted_profiles(p, rng.multinomial(256, np.full(256, 1/256), size=2000))
                    for p in data]
            point = [weighted_profiles(p, np.ones((1, 256))) for p in data]
            draws = differences(*boot)
            estimates = differences(*point)
            within, between = matched_covariance(*data, rng)
            arrays = dict(time=times[1:], **draws,
                reference_split_covariance_errors=within,
                generated_reference_covariance_errors=between,
                generated_full_covariance=full_covariance(data[0]),
                reference_full_covariance=full_covariance(data[1]))
            assert all(np.isfinite(v).all() for v in arrays.values())
            path = root/f"seed{seed}_anchor{row['anchor']}.npz"
            np.savez_compressed(path, **arrays)
            metrics = {k: dict(estimate=v[0].tolist(),
                ci95=np.quantile(draws[k], [.025, .975], axis=0).tolist())
                for k, v in estimates.items()}
            result['rows'].append(dict(seed=seed, anchor=row['anchor'], split=row['split'],
                state=row['state'], rng_seed=rng_seed, path=str(path), sha256=digest(path),
                source_sha256=row['sha256'], reference_sha256=row['reference_sha256'],
                source_report_sha256=report_hash, metrics=metrics,
                covariance_partition_mean_ratio=(between.mean(0)/within.mean(0)).tolist()))
            save(root/'summary.json', result)
    result['status'] = 'COMPLETE_MIDPOINT_DISTRIBUTION_AUDIT_NOT_CERTIFIED'
    save(root/'summary.json', result)


if __name__ == '__main__':
    run()
