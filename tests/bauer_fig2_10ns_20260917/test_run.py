import numpy as np
import pytest
from tests.bauer_fig2_10ns_20260917.run import counts,observables,HBAR_MEV_PS,KB_MEV_K
from tests.bauer_fig2_10ns_20260917.kernel import trajectory,drift


def test_physical_time_grid():
    assert counts(10000,.00025,.1)==(40000000,400)
    assert .00025/(HBAR_MEV_PS/50)<.02
    assert .11*50/KB_MEV_K==pytest.approx(63.82485,abs=1e-5)
    with pytest.raises(ValueError): counts(1,.003,.1)


def test_energy_total_not_mean_and_open_boundaries():
    s=np.zeros((1,100,3)); s[...,2]=1
    m,e,error=observables(s,.1)
    np.testing.assert_allclose(m,[[0,0,100]])
    assert e[0]==pytest.approx(-1.09)
    assert error[0]==0
    np.testing.assert_allclose(drift(s[0],.1,.1),0)


def test_fixed_seed_reproducibility_and_norm():
    s=np.zeros((100,3)); s[:,2]=1
    a,err=trajectory(s,71,.019,100,10,.1,.1,.11)
    b,_=trajectory(s,71,.019,100,10,.1,.1,.11)
    c,_=trajectory(s,72,.019,100,10,.1,.1,.11)
    np.testing.assert_array_equal(a,b)
    assert not np.array_equal(a,c)
    assert np.isfinite(a).all() and np.isfinite(err)
    np.testing.assert_allclose(np.linalg.norm(a,axis=-1),1,atol=1e-14)
