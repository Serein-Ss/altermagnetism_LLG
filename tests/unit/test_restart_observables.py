import numpy as np
import pytest
from tests.experiments.restart_observables import per_parent, matched_restart


def test_open_chain_and_parent_axis_preserved():
    # Open-chain dot mean is zero; a wrapped third bond would make it -1/3.
    frame = np.array([[0, 0, 1], [1, 0, 0], [0, 0, -1.]])
    paths = np.stack([np.stack([frame, frame]), np.stack([-frame, -frame])])
    result = per_parent(paths)
    np.testing.assert_array_equal(result['spatial_dot_lag1'], np.zeros((2, 2)))
    np.testing.assert_array_equal(result['increment_second_moment_lag1'], np.zeros((2, 1)))
    np.testing.assert_array_equal(result['anchor_dot'], np.ones((2, 2)))
    assert result['magnetization'].shape == (2, 2, 3)


def test_matched_anchors_not_matched_noise():
    r = np.zeros((2, 3, 2, 3)); r[..., 2] = 1
    g = r.copy(); g[:, 1:, :, 2] = -1
    result = matched_restart(g, r)
    np.testing.assert_array_equal(result['differences']['spatial_dot_lag1'], np.zeros((2, 3)))
    np.testing.assert_array_equal(result['differences']['increment_second_moment_lag1'], [[4, 0], [4, 0]])
    np.testing.assert_array_equal(result['differences']['magnetization'][..., 2], [[0, -2, -2]]*2)
    g[:, 0] *= -1
    with pytest.raises(ValueError, match='identical anchors'):
        matched_restart(g, r)


def test_invalid_spin_and_shape_rejected():
    with pytest.raises(ValueError):
        per_parent(np.zeros((2, 3, 2, 3)))
    with pytest.raises(ValueError):
        matched_restart(np.zeros((2, 3, 2, 3)), np.zeros((1, 3, 2, 3)))
