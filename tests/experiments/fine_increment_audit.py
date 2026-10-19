"""Recompute pilot statistics and retain conditional, exploratory contrasts."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.fine_increment_statistics import increments, moments, bootstrap


def run():
    root = Path('tests/results/fine_increment_statistics')
    source = root / 'summary.json'
    report = json.loads(source.read_text())
    raw_report = json.loads(Path('tests/results/fine_transition_pilot/summary.json').read_text())
    raw_rows = {(r['case'], r['anchor']): r for r in raw_report['rows']}
    if report['status'] != 'COMPLETE_FINITE_LAG_MOMENTS_NOT_CERTIFIED':
        raise ValueError('incomplete source')
    expected = {(c, a) for c in range(4) for a in range(12)}
    assert len(report['rows']) == 48
    assert {(r['case'], r['anchor']) for r in report['rows']} == expected
    lags = np.asarray(report['lags'])
    summaries = []; by_case = {}
    for row in report['rows']:
        key = row['case'], row['anchor']
        raw = raw_rows[key]
        assert digest(raw['path']) == row['source_sha256'] == raw['sha256']
        assert digest(row['path']) == row['sha256']
        with np.load(raw['path']) as d:
            ambient, tangent = increments(d['spins'], d['reduced_time'], d['spins'][0, 0], lags)
        weights = np.random.default_rng(row['bootstrap_seed']).multinomial(
            256, np.full(256, 1/256), size=report['bootstrap_replicates'])
        with np.load(row['path']) as d:
            assert all(np.isfinite(d[k]).all() for k in d.files)
            for label, delta in [('ambient', ambient), ('tangent', tangent)]:
                m, c = moments(delta, lags)
                bm, bt = bootstrap(delta, lags, weights)
                np.testing.assert_allclose(m, d[label+'_mean_rate'], atol=1e-12)
                np.testing.assert_allclose(c, d[label+'_covariance_rate'], atol=1e-12)
                np.testing.assert_allclose(np.quantile(bm, [.025, .975], axis=0), d[label+'_mean_rate_ci95'], atol=1e-12)
                np.testing.assert_allclose(np.quantile(bt, [.025, .975], axis=0), d[label+'_trace_rate_ci95'], atol=1e-12)
                np.testing.assert_allclose(bt.mean(-1), d[label+'_site_average_trace_draws'], atol=1e-12)
                # Independent covariance implementation; cross-site entries retained.
                for k, lag in enumerate(lags):
                    np.testing.assert_allclose(c[k], np.cov(delta[:, k].reshape(256, -1), rowvar=False)/lag, atol=1e-12)
            trace = np.trace(d['tangent_covariance_rate'], axis1=1, axis2=2)/25
            draws = d['tangent_site_average_trace_draws']
            by_case[key] = trace, draws
            summaries.append(dict(case=key[0], anchor=key[1], state=row['state'],
                tangent_site_average_trace_rate=trace.tolist(),
                trace_rate_ci95=np.quantile(draws, [.025, .975], axis=0).tolist(),
                median_component_mean_rate_ci_halfwidth=np.median(
                    (d['tangent_mean_rate_ci95'][1]-d['tangent_mean_rate_ci95'][0])/2, axis=(1, 2)).tolist(),
                lag_change_from_shortest=(trace-trace[0]).tolist(),
                lag_change_ci95=np.quantile(draws-draws[:, :1], [.025, .975], axis=0).tolist()))
    contrasts = []
    for anchor in range(12):
        ref, ref_draws = by_case[2, anchor]
        for case in [0, 1, 3]:
            point, draws = by_case[case, anchor]
            contrasts.append(dict(case=case, reference_case=2, anchor=anchor,
                trace_rate_difference=(point-ref).tolist(),
                ci95=np.quantile(draws-ref_draws, [.025, .975], axis=0).tolist()))
    dest = root / 'audit.json'
    if dest.exists():
        raise FileExistsError('preserve previous audit')
    save(dest, dict(status='COMPLETE_RECOMPUTATION_NOT_CERTIFICATION', source_sha256=digest(source),
        code_sha256=digest(__file__), lags=lags.tolist(), rows=summaries, contrasts=contrasts,
        scope='Pointwise exploratory percentile intervals, no equivalence margin or pass decision. '
        'Case contrasts use independent noise bootstrap draws, not paired physical noises. '
        'Lag contrasts reuse whole-path bootstrap draws. No across-anchor population inference. '
        'Trace only summarizes noise strength; full covariance and path law require separate validation.'))
    print(f'Verified {len(summaries)} groups; saved {len(contrasts)} numerical case contrasts to {dest}')


if __name__ == '__main__':
    run()
