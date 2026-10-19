import numpy as np
from tests.experiments.local_transition_baselines import random_walk, profiles, compare


def test_baseline_geometry_and_profiles():
    initial = np.tile([0., 0., 1.], (5, 1))
    a = random_walk(initial, 16, 9, .02, 10)
    np.testing.assert_array_equal(a, random_walk(initial, 16, 9, .02, 10))
    assert not np.array_equal(a, random_walk(initial, 16, 9, .02, 11))
    np.testing.assert_allclose(np.linalg.norm(a, axis=-1), 1, atol=1e-14)
    p = profiles(a)
    assert all(np.all(np.asarray(v)==0) for v in compare(p, p).values())
    frozen = profiles(np.broadcast_to(a[:, :1], a.shape))
    np.testing.assert_allclose(frozen['mean_increment'], 0)
    np.testing.assert_allclose(frozen['site_variance'], 0)
    np.testing.assert_allclose(p['site_variance'], np.sum((a-a.mean(0))**2, axis=(0, 3))/15)
