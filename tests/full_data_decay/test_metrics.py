import unittest
import numpy as np
from tests.bauer_p2_20260923.common import statistics, comparison
from tests.bauer_comprehensive_20260923.components import diagnostic_arrays
from .metrics import estimates, compare


class EstimandTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(93001)
        self.r = rng.normal(size=(8, 11, 16, 3))
        self.r /= np.linalg.norm(self.r, axis=-1, keepdims=True)
        self.g = self.r[::-1].copy()

    def test_population_unchanged(self):
        expected = {**statistics(self.r), **diagnostic_arrays(self.r)}
        for key, value in estimates(self.r, 'population').items():
            np.testing.assert_array_equal(value, expected[key])

    def test_empirical_direct_formulas(self):
        a = estimates(self.r, 'empirical')
        d = np.diff(self.r, axis=1); c = d-d.mean(0)
        np.testing.assert_allclose(a['increment_variance'], (c*c).sum(-1).mean((0, 2)))
        np.testing.assert_allclose(a['increment_cov'], np.einsum('btni,btnj->tnij', c, c)/8)
        np.testing.assert_allclose(a['increment_cross_covariance'], (c[:, 1:]*c[:, :-1]).sum(-1).mean((0, 2)))
        np.testing.assert_allclose(a['mz_var'], self.r[..., 2].mean(-1).var(0))
        p = estimates(self.r, 'population')
        for key in ('energy_var', 'mz_var', 'increment_variance'):
            np.testing.assert_allclose(a[key], p[key]*7/8)
        for key in ('energy_mean', 'mz_mean', 'spatial', 'two_time_spin_correlation'):
            np.testing.assert_array_equal(a[key], p[key])

    def test_population_bootstrap_regression(self):
        old = comparison(self.g, self.r, 10, 123)
        new = compare(self.g, self.r, 'population', 10, 123)['metrics']
        for key in old:
            for field in old[key]:
                np.testing.assert_array_equal(old[key][field], new[key][field])

    def test_empirical_reference_not_resampled(self):
        result = compare(self.g, self.r, 'empirical', 10, 123)['metrics']
        rng = np.random.default_rng(123); fixed = estimates(self.r, 'empirical')['increment_variance']
        diffs = []
        for _ in range(10):
            sample = self.g[rng.integers(8, size=8)]
            diffs.append((estimates(sample, 'population')['increment_variance']-fixed).mean())
        np.testing.assert_array_equal(result['increment_variance']['mean_difference_ci95'], np.quantile(diffs, [.025, .975]))

    def test_invalid(self):
        with self.assertRaises(ValueError): estimates(self.r, 'unknown')
        with self.assertRaises(ValueError): estimates(self.r[:1], 'population')


if __name__ == '__main__': unittest.main()
