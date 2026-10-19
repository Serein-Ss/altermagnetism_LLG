import numpy as np
import pytest
from tests.bauer_teacher_state_20260920.prepare import frame_indices


def test_physical_times():
    assert frame_indices(np.arange(101)*0.1, [0., 1., 5., 9.]) == [0, 10, 50, 90]
    with pytest.raises(ValueError, match='absent'):
        frame_indices(np.arange(11)*0.1, [5.])
