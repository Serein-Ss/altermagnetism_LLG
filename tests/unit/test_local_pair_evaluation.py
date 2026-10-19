import numpy as np
import pytest
from tests.experiments.local_pair_evaluation import development_groups


def test_reference_indices_and_no_silent_truncation():
    paths=np.zeros((64,3,2,3)); paths[...,2]=1
    paths[16:32,...,2]=-1; paths[48:64,...,2]=-1
    conditions=np.array([[.11,.1]]*32+[[.13,.1]]*32)
    initial,indices=development_groups(paths,conditions,.13)
    np.testing.assert_array_equal(indices,np.arange(32,64).reshape(2,16))
    np.testing.assert_array_equal(initial,paths[[32,48],0])
    with pytest.raises(ValueError): development_groups(paths[:-1],conditions[:-1],.13)
