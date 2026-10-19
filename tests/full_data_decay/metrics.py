"""Explicit fixed-empirical versus independent-population estimands, v2."""
import numpy as np
from tests.bauer_p2_20260923.common import statistics
from tests.bauer_comprehensive_20260923.components import diagnostic_arrays


def estimates(paths, kind):
    if kind not in ('population', 'empirical'):
        raise ValueError(kind)
    if len(paths) < 2:
        raise ValueError('At least two whole branches are required')
    values = {**statistics(paths), **diagnostic_arrays(paths)}
    if kind == 'empirical':
        for key in ('mz_var', 'energy_var', 'increment_cov',
                    'increment_variance', 'increment_cross_covariance'):
            values[key] = values[key] * ((len(paths)-1)/len(paths))
    return values


def compare(generated, reference, reference_kind, repeats=100, seed=93077):
    a, b = estimates(generated, 'population'), estimates(reference, reference_kind)
    rows = {k: dict(generated=a[k].tolist(), reference=b[k].tolist(),
                   rmse=float(np.sqrt(np.mean((a[k]-b[k])**2))),
                   mean_difference=float((a[k]-b[k]).mean())) for k in a}
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in a}
    for _ in range(repeats):
        ga = estimates(generated[rng.integers(len(generated), size=len(generated))], 'population')
        rb = b if reference_kind == 'empirical' else estimates(
            reference[rng.integers(len(reference), size=len(reference))], 'population')
        for k in a:
            boot[k].append(float((ga[k]-rb[k]).mean()))
    if repeats:
        for k in a:
            rows[k]['mean_difference_ci95'] = np.quantile(boot[k], [.025, .975]).tolist()
    return dict(version=2, reference_kind=reference_kind,
                uncertainty='whole-branch percentile bootstrap; fixed empirical reference is not resampled; CI is for mean difference, not RMSE or equivalence',
                repeats=repeats, seed=seed, metrics=rows)
