import numpy as np
import pytest
from tests.experiments.reference_initial_bank import assert_disjoint_initials


def test_float_precision_does_not_hide_initial_reuse():
    a=np.array([[[.1,.2,.3],[.4,.5,.6]]],dtype=np.float64)
    with pytest.raises(ValueError,match='overlap'):
        assert_disjoint_initials(a,dict(training=a.astype(np.float32)))
    b=a.copy();b[0,0,0]+=.01
    assert assert_disjoint_initials(b,dict(training=a))['candidate_initials']==1
    with pytest.raises(ValueError,match='duplicate'):
        assert_disjoint_initials(np.concatenate([a,a]),{})
    with pytest.raises(ValueError,match='shape'):
        assert_disjoint_initials(a,dict(training=np.zeros((1,3,3))))
