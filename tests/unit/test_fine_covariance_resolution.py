import numpy as np
from tests.experiments.fine_covariance_resolution import covariance, errors, split_covariances, LAGS


def test_covariance_and_split():
    delta = np.random.default_rng(1).normal(size=(16, 4, 5, 3))
    c = covariance(delta)
    for k, lag in enumerate(LAGS):
        np.testing.assert_allclose(c[k], np.cov(delta[:, k].reshape(16, -1), rowvar=False)/lag)
    a, b = split_covariances(delta, np.random.default_rng(2))
    order = np.random.default_rng(2).permutation(16)
    np.testing.assert_allclose(a, covariance(delta[order[:8]]))
    np.testing.assert_allclose(b, covariance(delta[order[8:]]))


def test_spatial_block_masks_and_rotation():
    c = np.zeros((4, 9, 9)); d = c.copy()
    d[:, :3, 6:] = 1; d[:, 6:, :3] = 1
    e = errors(c, d)
    np.testing.assert_allclose(e[:, 0], np.sqrt(18/81))
    np.testing.assert_allclose(e[:, 1], 0)
    np.testing.assert_allclose(e[:, 2], np.sqrt(18/54))
    np.testing.assert_allclose(e[:, 3], 0)
    q, _ = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))
    transform = np.kron(np.eye(3), q)
    np.testing.assert_allclose(errors(c, transform.T @ d @ transform), e, atol=1e-14)
