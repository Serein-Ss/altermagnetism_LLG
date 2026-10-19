import numpy as np
import pytest
from tests.experiments.fresh_reference_audit import path_energy


def test_energy_unbiased_and_preserves_joint_structure():
    a = np.array([[0.],[2.]])
    b = np.array([[4.],[6.]])
    assert path_energy(a,b) == 4
    assert path_energy(a,a) == -2
    np.testing.assert_allclose(path_energy(a,b),path_energy(b,a))
    same = np.tile([[1,1],[-1,-1]],(20,1))
    opposite = np.tile([[1,-1],[-1,1]],(20,1))
    assert path_energy(same,opposite) > .5
    with pytest.raises(ValueError):
        path_energy(a[:1],b)
