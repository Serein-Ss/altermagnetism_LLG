import numpy as np
from tests.experiments.director_midpoint_summary import full_covariance


def test_covariance_retains_cross_site_entries():
    p=np.random.default_rng(2).normal(size=(16,9,5,3))
    p[:,:,1]=p[:,:,0]
    c=full_covariance(p)
    for t in range(8):
        np.testing.assert_allclose(c[t],np.cov(p[:,t+1].reshape(16,-1),rowvar=False),atol=1e-14)
        np.testing.assert_allclose(c[t,:3,3:6],c[t,:3,:3],atol=1e-14)
