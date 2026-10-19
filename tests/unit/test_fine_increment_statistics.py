import numpy as np
import pytest
from tests.experiments.fine_increment_statistics import increments, moments, bootstrap


def test_moments_and_whole_path_bootstrap():
    rng = np.random.default_rng(1)
    delta = rng.normal(size=(7, 2, 4, 3))
    lags = np.array([.01, .02])
    mean, covariance = moments(delta, lags)
    np.testing.assert_allclose(mean, delta.mean(0) / lags[:, None, None])
    for k in range(2):
        np.testing.assert_allclose(covariance[k], np.cov(delta[:, k].reshape(7, -1), rowvar=False) / lags[k])
    indices = rng.integers(0, 7, size=(5, 7))
    weights = np.array([np.bincount(i, minlength=7) for i in indices])
    bm, bt = bootstrap(delta, lags, weights)
    for b, ids in enumerate(indices):
        m, c = moments(delta[ids], lags)
        np.testing.assert_allclose(bm[b], m, atol=1e-12)
        trace = np.diagonal(c, axis1=-2, axis2=-1).reshape(2, 4, 3).sum(-1)
        np.testing.assert_allclose(bt[b], trace, atol=1e-12)


def test_projection_rotation_and_rejection():
    rng = np.random.default_rng(2)
    spins = rng.normal(size=(6, 3, 4, 3))
    spins /= np.linalg.norm(spins, axis=-1, keepdims=True)
    spins[:, 0] = spins[0, 0]
    anchor = spins[0, 0]
    times = np.array([0., .01, .02])
    lags = times[1:]
    a, t = increments(spins, times, anchor, lags)
    np.testing.assert_allclose((t * anchor).sum(-1), 0, atol=1e-15)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    ar, tr = increments(spins @ q, times, anchor @ q, lags)
    np.testing.assert_allclose(ar, a @ q, atol=1e-15)
    np.testing.assert_allclose(tr, t @ q, atol=1e-15)
    with pytest.raises(ValueError, match='no interpolation'):
        increments(spins, times, anchor, [.015])
    broken = spins.copy(); broken[0, 1, 0, 0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        increments(broken, times, anchor, lags)
