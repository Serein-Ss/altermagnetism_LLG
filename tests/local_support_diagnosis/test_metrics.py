import numpy as np
import pytest
from .metrics import patches, nearest, bias


def test_open_edges_and_axial_alignment():
    rng = np.random.default_rng(17)
    a = rng.normal(size=(16, 3)); a /= np.linalg.norm(a, axis=-1, keepdims=True)
    p, m = patches(a, 2)
    assert m[0].tolist() == [False, False, True, True, True]
    assert m[-1].tolist() == [True, True, True, False, False]
    angle = .7
    rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    q, qm = patches(a @ rot.T, 2)
    assert np.max(nearest(q, qm, p, m)) < 3e-8
    assert np.isnan(nearest(q[:1], qm[:1], p[3:4], m[3:4])).all()


def test_bias_noise_correction_and_components():
    initial = np.array([[0., 0., 1.]])
    g = np.broadcast_to(initial, (4, 2, 1, 3)).copy()
    r = g.copy(); g[:, 1, 0, 0] = [-1, 1, -1, 1]
    result = bias(g, r)
    assert result['mean_rmse'] == 0
    assert result['corrected_mse'] < 0
    g[:, 1, 0, 2] += .3
    result = bias(g, r)
    assert result['longitudinal_rmse'] == pytest.approx(.3)
    assert result['transverse_rmse'] == 0
    r[:, 0, 0, 0] = .01
    with pytest.raises(ValueError, match='initial'):
        bias(g, r)
