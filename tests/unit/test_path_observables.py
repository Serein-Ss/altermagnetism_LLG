import numpy as np
import pytest
from tests.experiments.path_observables import path_observables


def smooth_paths():
    rng=np.random.default_rng(73)
    t=np.linspace(0,1,41)[None,:,None]
    r=np.linspace(0,2*np.pi,25)[None,None,:]
    phase=t*(rng.uniform(.5,2,(32,1,1))+r)
    return np.stack((np.sin(phase),np.zeros_like(phase),np.cos(phase)),axis=-1)


def test_time_scramble_changes_temporal_dependence():
    s=smooth_paths(); rng=np.random.default_rng(12)
    scrambled=s[:,np.r_[0,rng.permutation(np.arange(1,s.shape[1]))]]
    a=path_observables(s); b=path_observables(scrambled)
    assert np.allclose(a['magnetization_per_path'].mean(1),b['magnetization_per_path'].mean(1))
    assert b['increment_second_moment_lag1'].mean()>10*a['increment_second_moment_lag1'].mean()


def test_site_scramble_preserves_magnetization_but_breaks_space():
    s=smooth_paths(); scrambled=s[:,:,np.random.default_rng(14).permutation(s.shape[2])]
    a=path_observables(s); b=path_observables(scrambled)
    assert np.allclose(a['magnetization_per_path'],b['magnetization_per_path'])
    assert a['spatial_dot_lag1'].mean()-b['spatial_dot_lag1'].mean()>.1


def test_replicated_path_has_no_ensemble_diversity():
    s=smooth_paths(); a=path_observables(s)
    b=path_observables(np.repeat(s[:1],len(s),axis=0))
    assert a['spin_ensemble_variance'].max()>.01
    assert b['spin_ensemble_variance'].max()<1e-25
    assert b['magnetization_variance'].max()<1e-25


def test_exact_identities_and_invalid_input():
    s=smooth_paths(); a=path_observables(s)
    assert np.allclose(a['increment_second_moment_lag1'],2-2*a['temporal_dot_lag1'])
    assert np.allclose(a['spatial_spin_variance'],1-np.square(a['magnetization_per_path']).sum(-1).mean(0))
    s[1,0,0]=[1,0,0]
    with pytest.raises(ValueError,match='initial'):path_observables(s)
    with pytest.raises(ValueError,match='unit'):path_observables(s*2)
    with pytest.raises(ValueError,match='lag'):path_observables(smooth_paths(),temporal_lags=(41,))
