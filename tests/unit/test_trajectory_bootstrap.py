import numpy as np
import pytest
from tests.experiments.trajectory_bootstrap import bootstrap_weights,weighted_observables
from tests.experiments.path_observables import path_observables


def fixture():
    s=np.random.default_rng(10).normal(size=(5,41,15,3))
    s/=np.linalg.norm(s,axis=-1,keepdims=True);s[:,0]=s[0,0]
    return s


def test_weighted_moments_equal_explicit_whole_path_resampling():
    s=fixture();w=bootstrap_weights(5,8,37)
    np.testing.assert_array_equal(w,bootstrap_weights(5,8,37))
    obs=weighted_observables(s,w)
    for row in range(len(w)):
        repeated=np.repeat(np.arange(5),np.rint(w[row]*5).astype(int))
        ref=path_observables(s[repeated])
        assert set(obs)==set(ref)-{'magnetization_per_path'}
        for key in obs:np.testing.assert_allclose(obs[key][row],ref[key],atol=2e-14,rtol=2e-13)


def test_uniform_weights_and_single_path_collapse():
    s=fixture();w=np.array([[.2]*5,[1,0,0,0,0]])
    obs=weighted_observables(s,w);ref=path_observables(s)
    for key in obs:np.testing.assert_allclose(obs[key][0],ref[key],atol=2e-14)
    np.testing.assert_allclose(obs['magnetization_variance'][1],0,atol=2e-14)
    np.testing.assert_allclose(obs['spin_ensemble_variance'][1],0,atol=2e-14)
    for bad in (w*.5,w[:,:4],np.full((1,5),np.nan)):
        with pytest.raises(ValueError):weighted_observables(s,bad)
    with pytest.raises(ValueError):bootstrap_weights(1,10,1)
