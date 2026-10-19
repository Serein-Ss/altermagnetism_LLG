import numpy as np
from tests.experiments.midpoint_spatial_mean import spatial_error
from tests.experiments.midpoint_mean_resolution import corrected_mean_mse


def test_spatial_aggregation_and_localization():
    rng = np.random.default_rng(18)
    a = rng.normal(size=(16, 4, 25, 3))
    b = rng.normal(size=a.shape)
    difference, raw, noise, corrected = spatial_error(a, b)
    expected = corrected_mean_mse(a, b)
    for x, y in zip([raw, noise, corrected], expected):
        np.testing.assert_allclose(x.mean(-1), y)
    b = a.copy()
    b[:, 1:, 12, 0] += 3
    difference, raw, _, _ = spatial_error(a, b)
    np.testing.assert_allclose(raw[:, 12], 3)
    np.testing.assert_allclose(np.delete(raw, 12, axis=1), 0)
