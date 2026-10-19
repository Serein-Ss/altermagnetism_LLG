import numpy as np
import pytest
from tests.bauer_teacher_state_20260920.report import moments


def test_known_shift_and_covariance():
    b = np.zeros((4, 2, 2, 3)); a = b.copy()
    a[:, 1, :, 2] = 0.3
    r = moments(a, b)
    assert r['mean_increment']['corrected_mse'] == pytest.approx(0.03)
    assert r['signed_mz_increment'] == pytest.approx(0.3)
    assert r['local_covariance_relative'] == 0
    with pytest.raises(ValueError, match='equal branch'):
        moments(a[:2], b)


def test_covariance_zero_for_identical_ensembles():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(64, 2, 4, 3)); a[:, 0] = 0
    r = moments(a, a)
    assert r['local_covariance_relative'] == 0
    assert r['signed_power_relative'] == pytest.approx(0)
