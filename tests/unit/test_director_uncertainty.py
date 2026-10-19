import numpy as np
from tests.experiments.director_uncertainty import weighted_profiles


def test_path_weights_match_explicit_resampling():
    rng = np.random.default_rng(2)
    paths = rng.normal(size=(12, 9, 5, 3))
    paths[:, 0] = paths[0, 0]
    ids = rng.integers(0, 12, 12)
    weights = np.bincount(ids, minlength=12)[None]
    result = weighted_profiles(paths, weights)
    p = paths[ids]; delta = p-p[:, :1]
    mean, variance, bond, temporal = result
    np.testing.assert_allclose(mean[0], delta.mean(0)[1:])
    np.testing.assert_allclose(variance[0], p.var(0, ddof=1).sum(-1).mean(-1)[1:])
    np.testing.assert_allclose(bond[0], (p[:, :, :-1]*p[:, :, 1:]).sum(-1).mean(0)[1:])
    centered = p-p.mean(0)
    expected = (centered[:, 1:]*centered[:, :-1]).sum(-1).sum(0).mean(-1)/11
    np.testing.assert_allclose(temporal[0], expected, atol=1e-14)
