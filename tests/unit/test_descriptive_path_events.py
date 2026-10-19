import numpy as np
import pytest
from tests.experiments.descriptive_path_events import sign_events


def test_endpoint_crossing_and_observation_time_are_distinct():
    result = sign_events([[1, .2, .1], [1, -.2, .1], [1, .2, -.1],
                          [1, 0, .1], [1, -.1, 0]], [0, 80, 160])
    np.testing.assert_array_equal(result['negative_endpoint'], [0, 0, 1, 0, 0])
    np.testing.assert_array_equal(result['crossed_zero'], [0, 1, 1, 0, 1])
    np.testing.assert_allclose(result['first_observed_negative_time'],
                               [np.nan, 80, 160, np.nan, 80], equal_nan=True)
    np.testing.assert_array_equal(result['observation_end'], [160]*5)
    assert 'committed_switch' not in result


@pytest.mark.parametrize('mz,time', [
    ([[0, -.1]], [0, 1]), ([[1, np.nan]], [0, 1]),
    ([[1, 2]], [0, 1]), ([[1, -.1]], [1, 1]),
    ([[1, -.1]], [0]), ([[1]], [0]), ([], [0, 1]),
])
def test_invalid_inputs_rejected(mz, time):
    with pytest.raises(ValueError):
        sign_events(mz, time)
