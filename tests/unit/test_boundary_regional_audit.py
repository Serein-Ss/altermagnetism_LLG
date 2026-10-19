import numpy as np

from tests.experiments.boundary_regional_audit import regional_errors


def test_regions_partition_mean_and_bond_errors():
    rng = np.random.default_rng(42)
    a, b = rng.normal(size=(2, 8, 25, 3))
    result = regional_errors(a, b)
    mean_mse = ((a.mean(0) - b.mean(0)) ** 2).mean()
    bond_a = (a[:, :-1] * a[:, 1:]).sum(-1).mean(0)
    bond_b = (b[:, :-1] * b[:, 1:]).sum(-1).mean(0)
    np.testing.assert_allclose(
        (6 * result['ends']['mean_rmse'] ** 2 + 19 * result['interior']['mean_rmse'] ** 2) / 25,
        mean_mse)
    np.testing.assert_allclose(
        (4 * result['ends']['bond_rmse'] ** 2 + 20 * result['interior']['bond_rmse'] ** 2) / 24,
        ((bond_a - bond_b) ** 2).mean())
    assert all(value == 0 for region in regional_errors(a, a).values() for value in region.values())
