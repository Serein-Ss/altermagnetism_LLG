import numpy as np
from tests.experiments.midpoint_mean_resolution import corrected_mean_mse


def test_mean_noise_subtraction_and_no_clipping():
    a = np.zeros((4, 2, 1, 3))
    a[:, 1, 0, 0] = [-3, -1, 1, 3]
    b = a.copy()
    raw, noise, corrected = corrected_mean_mse(a, b)
    np.testing.assert_allclose(raw, 0)
    np.testing.assert_allclose(noise, 10/9)
    np.testing.assert_allclose(corrected, -10/9)
    b[:, 1, 0, 1] += 2
    raw, noise, corrected = corrected_mean_mse(a, b)
    np.testing.assert_allclose(raw, 4/3)
    np.testing.assert_allclose(corrected, 2/9)
