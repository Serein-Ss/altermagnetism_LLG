import numpy as np
from tests.experiments.coverage_bond_decomposition import components


def test_zero_mean_correlated_spins_and_exact_denominator():
    p=np.zeros((2,1,2,3)); p[0,:,:,2]=1; p[1,:,:,2]=-1
    c=components(p)
    np.testing.assert_array_equal(c['total'],1)
    np.testing.assert_array_equal(c['mean_product'],0)
    np.testing.assert_array_equal(c['connected'],1)


def test_deterministic_spins_and_random_identity():
    rng=np.random.default_rng(54)
    p=rng.normal(size=(7,5,6,3)); p/=np.linalg.norm(p,axis=-1,keepdims=True)
    c=components(p)
    np.testing.assert_allclose(c['total'],c['mean_product']+c['connected'],atol=1e-14)
    c=components(np.repeat(p[:1],7,axis=0))
    np.testing.assert_allclose(c['connected'],0,atol=1e-28)
