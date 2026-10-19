import numpy as np
from tests.experiments.director_budget_analysis import paired_statistics


def test_identical_arms_have_exact_zero_paired_changes():
    rng = np.random.default_rng(14)
    a = rng.normal(size=(16, 4, 3, 3))
    r = rng.normal(size=a.shape)
    a[:, 0] = r[:, 0] = 0
    arrays, metrics = paired_statistics([a, a.copy(), r], rng, replicates=100)
    for key, metric in metrics['paired_change'].items():
        np.testing.assert_array_equal(arrays['paired_change_'+key], 0)
        np.testing.assert_array_equal(metric['estimate'], 0)
        np.testing.assert_array_equal(metric['ci95'], 0)
