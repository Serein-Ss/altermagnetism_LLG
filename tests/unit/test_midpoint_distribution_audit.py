import numpy as np
import pytest
from tests.experiments.midpoint_distribution_audit import covariance_errors, matched_covariance
from tests.experiments.director_convergence_summary import full_covariance


def test_cross_site_covariance_and_matched_counts():
    rng = np.random.default_rng(8)
    p = rng.normal(size=(12, 3, 2, 3))
    q = rng.normal(size=p.shape)
    c = full_covariance(p)
    np.testing.assert_allclose(c[0], np.cov(p[:, 1].reshape(12, 6), rowvar=False))
    a = c.copy()
    a[:, :3, 3:] += 2
    a[:, 3:, :3] += 2
    e = covariance_errors(a, c)
    np.testing.assert_allclose(e[:, 1], 0)
    np.testing.assert_allclose(e[:, 2:], 2)
    within, between = matched_covariance(p, q, np.random.default_rng(9), repeats=1)
    r = np.random.default_rng(9)
    ri = r.permutation(12)
    gi = r.permutation(12)[:6]
    ra = full_covariance(q[ri[:6]])
    np.testing.assert_allclose(within[0], covariance_errors(ra, full_covariance(q[ri[6:]])))
    np.testing.assert_allclose(between[0], covariance_errors(full_covariance(p[gi]), ra))
    with pytest.raises(ValueError):
        matched_covariance(p[:-1], q[:-1], rng)
