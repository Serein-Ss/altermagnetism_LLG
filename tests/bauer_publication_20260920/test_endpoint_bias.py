import numpy as np
import pytest
from tests.bauer_publication_20260920.endpoint_bias import compare


def test_signed_bias_and_sampling_error():
    a = np.zeros((2, 3, 4, 3)); b = a.copy()
    a[:, 1:, :, 2] = 0.25
    result = compare(a, b)
    assert result['signed_bias'] == [0., 0.25, 0.25]
    assert result['branch_sampling_se'] == [0., 0., 0.]
    b[0, 0, 0, 2] = 1.
    with pytest.raises(ValueError, match='initial state'):
        compare(a, b)


def test_branch_standard_error():
    a = np.zeros((2, 2, 1, 3)); b = a.copy()
    a[:, 1, 0, 2] = [-1., 1.]
    assert compare(a, b)['branch_sampling_se'] == [0., 1.]
