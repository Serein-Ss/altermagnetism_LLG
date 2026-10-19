"""Paired budget comparison with independent whole-path reference bootstrap."""
import json
from pathlib import Path
import numpy as np
import torch
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.director_uncertainty import weighted_profiles, differences
from tests.experiments.director_convergence_summary import full_covariance


def paired_statistics(data, rng, replicates=2000):
    n = len(data[0])
    assert all(len(p) == n for p in data)
    wg = rng.multinomial(n, np.full(n, 1/n), size=replicates)
    wr = rng.multinomial(n, np.full(n, 1/n), size=replicates)
    boot = [weighted_profiles(p, w) for p, w in zip(data, [wg, wg, wr])]
    point = [weighted_profiles(p, np.ones((1, n))) for p in data]
    arrays, metrics = {}, {}
    for i, label in enumerate(['6000', '24000']):
        draws = differences(boot[i], boot[2])
        estimates = differences(point[i], point[2])
        metrics[label] = {}
        for key, value in draws.items():
            arrays[label+'_'+key] = value
            metrics[label][key] = dict(estimate=estimates[key][0].tolist(),
                ci95=np.quantile(value, [.025, .975], axis=0).tolist())
        arrays[label+'_variance_ratio'] = point[i][1][0]/point[2][1][0]
    metrics['paired_change'] = {}
    for key in metrics['6000']:
        draws = arrays['24000_'+key]-arrays['6000_'+key]
        arrays['paired_change_'+key] = draws
        metrics['paired_change'][key] = dict(
            estimate=(np.array(metrics['24000'][key]['estimate'])-
                      np.array(metrics['6000'][key]['estimate'])).tolist(),
            ci95=np.quantile(draws, [.025, .975], axis=0).tolist())
    return arrays, metrics


def run():
    reports = {}
    for seed in [11, 22, 33]:
        pair = []
        for budget in [6000, 24000]:
            path = Path(f'tests/results/director_budget/{budget}/{seed}/summary.json')
            report = json.loads(path.read_text())
            assert report['status'] == 'COMPLETE_TRAINING_AND_GENERATION_NOT_CERTIFIED'
            assert len(report['rows']) == 12 and {r['anchor'] for r in report['rows']} == set(range(12))
            assert report['sampler'] == 'projected_midpoint' and report['config']['flow_steps'] == 32
            assert report['config']['updates'] == budget
            assert digest(report['checkpoint']) == report['checkpoint_sha256']
            assert digest(report['training_rng_at6000']) == report['training_rng_at6000_sha256']
            pair.append((report, digest(path)))
        a, b = [p[0] for p in pair]
        assert {k for k in a['config'] if a['config'][k] != b['config'][k]} == {'updates'}
        assert a['source_sha256'] == b['source_sha256']
        assert a['train_parents'] == b['train_parents']
        assert a['development_parents'] == b['development_parents']
        assert torch.equal(torch.load(a['training_rng_at6000'], weights_only=True),
                           torch.load(b['training_rng_at6000'], weights_only=True))
        reports[seed] = pair
    root = Path('tests/results/director_budget_analysis')
    root.mkdir(exist_ok=False)
    result = dict(status='RUNNING', rows=[], replicates=2000,
        code_sha256={p: digest(p) for p in [__file__,
            'tests/experiments/director_uncertainty.py',
            'tests/experiments/director_convergence_summary.py']},
        scope='Fixed checkpoints/anchors, exploratory paired whole-path bootstrap between budgets, '
        'independent reference indices. Pointwise intervals, not simultaneous or equivalence tests. '
        'Training references reused in fitting; development reused for design. Three training seeds '
        'are blocks, not 36 independent training replicates. Changes include cosine schedule duration. '
        'Variance-ratio and full-covariance errors are descriptive. First adjacent-time connected covariance is zero.')
    for seed, pair in reports.items():
        indexed = [{r['anchor']: r for r in p[0]['rows']} for p in pair]
        for anchor in range(12):
            rows = [index[anchor] for index in indexed]
            for key in ['latent_seed', 'generation_batch', 'reference_path', 'reference_sha256', 'split', 'state']:
                assert rows[0][key] == rows[1][key]
            data = []
            sources = [(r['path'], r['sha256'], 1) for r in rows]
            sources.append((rows[0]['reference_path'], rows[0]['reference_sha256'], 4))
            for path, sha, stride in sources:
                assert digest(path) == sha
                with np.load(path) as z:
                    p = z['spins'][:, ::stride].astype(float)
                    t = z['reduced_time'][::stride]
                assert p.shape == (256, 9, 25, 3) and np.isfinite(p).all()
                assert np.max(abs(np.linalg.norm(p, axis=-1)-1)) < 1e-5
                if data:
                    np.testing.assert_allclose(t, times, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(p[:, 0], data[0][:, 0], atol=2e-7, rtol=0)
                data.append(p)
                times = t
            rng_seed = 992000000+seed*1000+anchor
            arrays, metrics = paired_statistics(data, np.random.default_rng(rng_seed))
            arrays['time'] = times[1:]
            rc = full_covariance(data[2])
            arrays['reference_full_covariance'] = rc
            for label, p in zip(['6000', '24000'], data[:2]):
                cov = full_covariance(p)
                arrays[label+'_full_covariance'] = cov
                arrays[label+'_full_covariance_rmse'] = np.sqrt(np.mean((cov-rc)**2, axis=(1, 2)))
            assert all(np.isfinite(v).all() for v in arrays.values())
            path = root/f'seed{seed}_anchor{anchor}.npz'
            np.savez_compressed(path, **arrays)
            result['rows'].append(dict(seed=seed, anchor=anchor, split=rows[0]['split'],
                state=rows[0]['state'], rng_seed=rng_seed, path=str(path), sha256=digest(path),
                source_report_sha256=[p[1] for p in pair], metrics=metrics))
            save(root/'summary.json', result)
    result['status'] = 'COMPLETE_BUDGET_COMPARISON_NOT_CERTIFIED'
    save(root/'summary.json', result)


if __name__ == '__main__':
    run()
