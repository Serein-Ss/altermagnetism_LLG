import numpy as np
from tests.paper_work.build_figures import (log_increment, local_features,
                                          probability_interval, arrival, observables, grouped_variance)


def test_spherical_increment_and_constant_observables():
    p = np.zeros((5, 3, 4, 3)); p[..., 2] = 1
    np.testing.assert_allclose(log_increment(p), 0)
    ob = observables(p)
    np.testing.assert_allclose(ob['variance'], 0)
    np.testing.assert_allclose(ob['bond'], 1)
    angle = .02
    p[:, 1, :, 0] = np.sin(angle); p[:, 1, :, 2] = np.cos(angle)
    np.testing.assert_allclose(log_increment(p)[..., 0], angle)
    np.testing.assert_allclose(log_increment(p)[..., 2], 0, atol=1e-12)


def test_features_respect_open_boundary():
    x = np.zeros((2, 4, 3)); x[..., 2] = 1
    f = local_features(x)
    assert f.shape == (2, 4, 6)
    np.testing.assert_allclose(f[:, 0, 1], 0)
    np.testing.assert_allclose(f[:, -1, 2], 0)
    np.testing.assert_allclose(f[:, 1:-1, 1:3], 1)


def test_wilson_zero_is_not_certified_zero_probability():
    lo, hi = probability_interval(0, 128)
    assert abs(lo) < 1e-12 and .02 < hi < .04
    lo, hi = probability_interval(128, 128)
    assert .96 < lo < .98 and abs(hi-1) < 1e-12


def test_core_arrival_retains_unresolved_and_initial_membership():
    mz = np.array([[0, 0, 0], [0, -.8, -.9], [.9, .9, -.9]])
    labels, t = arrival(mz, np.array([0, .1, .2]), -.7, .7, 0)
    np.testing.assert_array_equal(labels, [0, -1, 1])
    assert np.isinf(t[0]) and t[1] == .1 and t[2] == 0
    labels, t = arrival(mz, np.array([0, .1, .2]), -.7, .7, .1)
    np.testing.assert_array_equal(labels, [0, -1, 1])
    assert t[1] == .2 and t[2] == .1


def test_historical_variance_does_not_mix_initials():
    p=np.zeros((2,4,3,5,3));p[0,...,2]=1;p[1,...,2]=-1
    np.testing.assert_allclose(grouped_variance(p),0)
    assert p.reshape(8,3,5,3).var(0).sum()>0
