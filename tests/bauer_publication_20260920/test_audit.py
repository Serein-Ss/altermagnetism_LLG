import numpy as np
import pytest
from tests.bauer_publication_20260920.audit import mean_bias_squared, phase_statistics, initial_features


def test_unbiased_mean_error_in_expectation():
    rng = np.random.default_rng(9)
    a = rng.normal(size=(16, 20000))
    b = rng.normal(size=(32, 20000))
    assert abs(mean_bias_squared(a, b)['corrected_mse']) < .003
    assert abs(mean_bias_squared(a+.3, b)['corrected_mse']-.09) < .004


def test_does_not_clip_negative_estimate():
    a = np.array([[-1.], [1.]])
    assert mean_bias_squared(a, a)['corrected_mse'] == -2.
    with pytest.raises(ValueError):
        mean_bias_squared(a[:1], a)


def test_seam_index():
    d = np.ones((1, 30, 1, 3))
    d[:, ::10] = 2
    x = np.concatenate([np.zeros((1, 1, 1, 3)), np.cumsum(d, axis=1)], axis=1)
    result = phase_statistics(x)
    assert result['increment_power'] == [12.] + [3.]*9
    assert result['adjacent_increment_dot'][9] == 6.
    assert result['adjacent_increment_dot'][1] == 3.


def test_initial_features_are_configuration_level():
    x = np.zeros((4, 16, 3)); x[..., 2] = 1
    values = initial_features(x)
    assert all(v.shape == (4,) for v in values.values())
    assert np.all(values['fraction_sz_below_half'] == 0)
