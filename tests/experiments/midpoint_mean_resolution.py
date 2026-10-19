"""Mean-increment errors against matched-count reference sampling noise."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest, save


def mean_error(a, b):
    a = a-a[:, :1]
    b = b-b[:, :1]
    return np.mean((a.mean(0)-b.mean(0))**2, axis=(1, 2))[1:]


def corrected_mean_mse(a, b):
    raw = mean_error(a, b)
    a = a-a[:, :1]
    b = b-b[:, :1]
    noise = (a.var(0, ddof=1)/len(a)+b.var(0, ddof=1)/len(b)).mean((1, 2))[1:]
    return raw, noise, raw-noise


def run():
    source = Path('tests/results/midpoint_distribution_audit/summary.json')
    audit = json.loads(source.read_text())
    assert audit['status'] == 'COMPLETE_MIDPOINT_DISTRIBUTION_AUDIT_NOT_CERTIFIED'
    assert len(audit['rows']) == 36
    root = Path('tests/results/midpoint_mean_resolution')
    root.mkdir(exist_ok=False)
    result = dict(status='RUNNING', rows=[], source_sha256=digest(source),
        code_sha256=digest(__file__),
        scope='Exploratory fixed-anchor analysis. Raw squared mean error minus unbiased '
        'finite-ensemble mean-noise estimate; retain negative values, no clipping or square '
        'root of corrected MSE. Assumes independent paths within and between ensembles. '
        'Training references reused in fitting: correction there is descriptive only. '
        '100 repeated reference partitions are not independent experiments, CIs or hypothesis tests.')
    for row in audit['rows']:
        seed, anchor = row['seed'], row['anchor']
        report = json.loads(Path(f'tests/results/director_midpoint_sampling/{seed}/summary.json').read_text())
        item = next(r for r in report['rows'] if r['anchor'] == anchor and r['steps'] == 32)
        data = []
        for path, sha, stride in [(item['path'], row['source_sha256'], 1),
                (item['reference_path'], row['reference_sha256'], 4)]:
            assert digest(path) == sha
            with np.load(path) as z:
                data.append(z['spins'][:, ::stride].astype(float))
                times = z['reduced_time'][::stride]
        g, r = data
        assert g.shape == r.shape == (256, 9, 25, 3)
        raw, noise, corrected = corrected_mean_mse(g, r)
        rng_seed = 991000000+seed*1000+anchor
        rng = np.random.default_rng(rng_seed)
        within, between = [], []
        for _ in range(100):
            ri = rng.permutation(256)
            gi = rng.permutation(256)[:128]
            within.append(mean_error(r[ri[:128]], r[ri[128:]]))
            between.append(mean_error(g[gi], r[ri[:128]]))
        within, between = np.asarray(within), np.asarray(between)
        assert np.isfinite(corrected).all() and np.isfinite(between).all()
        path = root/f'seed{seed}_anchor{anchor}.npz'
        np.savez_compressed(path, time=times[1:], raw_mse=raw,
            estimated_mean_noise=noise, corrected_mse=corrected,
            reference_split_mse=within, generated_reference_mse=between)
        result['rows'].append(dict(seed=seed, anchor=anchor, split=row['split'], state=row['state'],
            rng_seed=rng_seed, path=str(path), sha256=digest(path),
            raw_mse=raw.tolist(), estimated_mean_noise=noise.tolist(), corrected_mse=corrected.tolist(),
            matched_partition_mse_ratio=(between.mean(0)/within.mean(0)).tolist()))
        save(root/'summary.json', result)
    result['status'] = 'COMPLETE_MEAN_RESOLUTION_NOT_CERTIFIED'
    save(root/'summary.json', result)


if __name__ == '__main__':
    run()
