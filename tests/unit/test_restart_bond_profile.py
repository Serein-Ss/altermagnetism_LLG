import numpy as np
import pytest
from tests.experiments.restart_bond_profile import bond_profile


def test_open_bonds_and_no_spatial_or_parent_pooling():
    r=np.zeros((2,3,4,3)); r[...,2]=1
    g=r.copy(); g[0,1:,0,2]=-1
    v=bond_profile(g,r)
    assert v['difference'].shape==(2,3,3)
    np.testing.assert_array_equal(v['difference'][0,1:],np.array([[-2,0,0],[-2,0,0]]))
    assert not v['difference'][1].any()


def test_anchor_mismatch_rejected():
    r=np.zeros((2,3,4,3)); r[...,2]=1
    g=r.copy(); g[0,0,0,2]=-1
    with pytest.raises(ValueError): bond_profile(g,r)
