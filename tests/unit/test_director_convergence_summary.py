import numpy as np
from tests.experiments.director_convergence_summary import full_covariance


def test_covariance_against_numpy():
    p=np.random.default_rng(1).normal(size=(16,9,5,3))
    cov=full_covariance(p)
    for t in range(8):
        np.testing.assert_allclose(cov[t],np.cov(p[:,t+1].reshape(16,-1),rowvar=False),atol=1e-14)
