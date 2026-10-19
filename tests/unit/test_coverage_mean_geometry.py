import numpy as np
import pytest
from tests.experiments.coverage_mean_geometry import geometry


def test_amplitude_only_and_direction_only():
    reference=np.zeros((2,3,3)); reference[:,:,2]=1
    g=geometry(reference*.8,reference)
    np.testing.assert_allclose(g['direction_contribution'],0,atol=1e-14)
    np.testing.assert_allclose(g['magnitude_contribution'],-.36)
    mean=reference.copy(); mean[:,1]=[1,0,0]
    g=geometry(mean,reference)
    np.testing.assert_allclose(g['magnitude_contribution'],0,atol=1e-14)
    np.testing.assert_allclose(g['direction_contribution'],-1)


def test_exchange_antisymmetry_and_undefined_direction():
    rng=np.random.default_rng(91)
    a=rng.normal(size=(4,5,3)); b=rng.normal(size=a.shape)
    ga=geometry(a,b); gb=geometry(b,a)
    for key in ['magnitude_contribution','direction_contribution','mean_bond_error']:
        np.testing.assert_allclose(ga[key],-gb[key],atol=1e-14)
    with pytest.raises(ValueError): geometry(a,np.zeros_like(b))
