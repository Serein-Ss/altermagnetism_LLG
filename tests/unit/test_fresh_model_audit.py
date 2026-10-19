import numpy as np
from tests.experiments.fresh_model_audit import temporal_covariance, sampler_statistics


def test_temporal_covariance_matches_explicit_sample_covariance():
    p = np.random.default_rng(9).normal(size=(16,5,3,3))
    expected = sum(np.cov(p[:,:,i,j],rowvar=False) for i in range(3) for j in range(3))/3
    np.testing.assert_allclose(temporal_covariance(p),expected)


def test_labels_and_paired_identical_samplers():
    rng = np.random.default_rng(12)
    p = rng.normal(size=(16,5,3,3))
    r = rng.normal(size=p.shape)
    p[:,0] = r[:,0] = 0
    arrays, metrics = sampler_statistics([p,p.copy(),r],rng,replicates=20)
    assert set(metrics) == {'32','64','paired_change'}
    assert arrays['32_mean_rmse'].shape == (20,4)
    np.testing.assert_array_equal(arrays['paired_change_mean_rmse'],0)
