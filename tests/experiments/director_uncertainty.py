"""Conditional whole-path bootstrap; exploratory pointwise intervals only."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest, save


def weighted_profiles(paths, weights):
    p = np.asarray(paths, dtype=np.float64)
    n, frames, sites, _ = p.shape
    if n < 2 or weights.shape[1] != n or not np.all(weights.sum(1) == n):
        raise ValueError('invalid path weights')
    delta = p-p[:, :1]
    mean = (weights @ delta.reshape(n, -1)/n).reshape(-1, frames, sites, 3)
    second = (weights @ (delta*delta).sum(-1).reshape(n, -1)/n).reshape(-1, frames, sites)
    variance = ((second-(mean*mean).sum(-1))*n/(n-1)).mean(-1)
    bond = (p[:, :, :-1]*p[:, :, 1:]).sum(-1)
    bond = (weights @ bond.reshape(n, -1)/n).reshape(-1, frames, sites-1)
    temporal = (delta[:, 1:]*delta[:, :-1]).sum(-1)
    temporal = (weights @ temporal.reshape(n, -1)/n).reshape(-1, frames-1, sites)
    temporal = ((temporal-(mean[:, 1:]*mean[:, :-1]).sum(-1))*n/(n-1)).mean(-1)
    return mean[:, 1:], variance[:, 1:], bond[:, 1:], temporal


def differences(candidate, reference):
    m, v, b, t = candidate; rm, rv, rb, rt = reference
    return dict(mean_rmse=np.sqrt(np.mean((m-rm)**2, axis=(-2, -1))),
                variance_difference=v-rv, bond_rmse=np.sqrt(np.mean((b-rb)**2, axis=-1)),
                temporal_connected_difference=t-rt)


def run():
    root = Path('tests/results/director_uncertainty')
    root.mkdir(exist_ok=False)
    summary = dict(status='RUNNING', rows=[], replicates=2000, seed=980000000,
        code_sha256=digest(__file__),
        scope='Whole-path percentile bootstrap, paired latent indices between arms, independent reference indices. '
        'Fixed initial states and checkpoints. Pointwise intervals, not simultaneous, no equivalence or pass decision. '
        'RMSE bootstrap not calibrated for zero-error testing. Train reference reused from training. '
        'Temporal connected covariance is between adjacent saved frames, averaged over sites; not complete temporal law.')
    for seed in [11, 22, 33]:
        reports = {}
        for arm in ['control', 'director']:
            path = Path(f'tests/results/director_controlled/{arm}/{seed}/summary.json')
            report = json.loads(path.read_text())
            assert report['status'] == 'COMPLETE_TRAINING_AND_GENERATION_NOT_CERTIFIED'
            assert len(report['rows']) == 12
            reports[arm] = {r['anchor']: r for r in report['rows']}
        assert set(reports['control']) == set(reports['director']) == set(range(12))
        for anchor in range(12):
            rows = [reports[a][anchor] for a in ['control', 'director']]
            assert rows[0]['latent_seed'] == rows[1]['latent_seed']
            assert rows[0]['generation_batch'] == rows[1]['generation_batch']
            assert rows[0]['reference_sha256'] == rows[1]['reference_sha256']
            data = []
            for row in rows:
                assert digest(row['path']) == row['sha256']
                with np.load(row['path']) as d:
                    data.append(d['spins'].astype(float)); times = d['reduced_time']
            assert digest(rows[0]['reference_path']) == rows[0]['reference_sha256']
            with np.load(rows[0]['reference_path']) as d:
                data.append(d['spins'][:, ::4]); rt = d['reduced_time'][::4]
            np.testing.assert_allclose(times, rt, atol=1e-12)
            for p in data:
                assert p.shape == (256, 9, 25, 3) and np.isfinite(p).all()
                np.testing.assert_allclose(p[:, 0], data[-1][:, 0], atol=2e-7, rtol=0)
            rng = np.random.default_rng(980000000+seed*1000+anchor)
            wg = rng.multinomial(256, np.full(256, 1/256), size=2000)
            wr = rng.multinomial(256, np.full(256, 1/256), size=2000)
            boot = [weighted_profiles(p, w) for p, w in zip(data, [wg, wg, wr])]
            point = [weighted_profiles(p, np.ones((1, 256))) for p in data]
            arrays = {'time': times[1:]}; metrics = {}
            for i, arm in enumerate(['control', 'director']):
                stats = differences(boot[i], boot[2]); estimates = differences(point[i], point[2])
                metrics[arm] = {}
                for key, draws in stats.items():
                    arrays[arm+'_'+key] = draws
                    metrics[arm][key] = dict(estimate=estimates[key][0].tolist(), ci95=np.quantile(draws, [.025, .975], axis=0).tolist())
            for key in metrics['control']:
                draws = arrays['director_'+key]-arrays['control_'+key]
                arrays['paired_change_'+key] = draws
            # Retain full cross-site covariance at every saved nonzero time.
            for label, p in zip(['control', 'director', 'reference'], data):
                centered = p[:, 1:]-p[:, 1:].mean(0)
                flat = centered.reshape(256, 8, 75).transpose(1, 0, 2)
                arrays[label+'_full_covariance'] = flat.transpose(0, 2, 1) @ flat/255
            assert all(np.isfinite(v).all() for v in arrays.values())
            path = root/f'seed{seed}_anchor{anchor}.npz'
            np.savez_compressed(path, **arrays)
            summary['rows'].append(dict(seed=seed, anchor=anchor, split=rows[0]['split'], state=rows[0]['state'],
                path=str(path), sha256=digest(path), metrics=metrics,
                source_sha256=[r['sha256'] for r in rows], reference_sha256=rows[0]['reference_sha256']))
            save(root/'summary.json', summary)
    summary['status'] = 'COMPLETE_POINTWISE_UNCERTAINTY_NOT_CERTIFIED'
    save(root/'summary.json', summary)


if __name__ == '__main__':
    run()
