import numpy as np
import pytest
from tests.bauer_publication_20260920.temporal_floor import paired_comparisons


def paths():
    r = np.random.default_rng(6).normal(size=(32, 11, 8, 3))
    r[:, 0] = 0
    return r


def test_matched_sizes_and_reference_direction():
    r = paths()
    out = paired_comparisons(r[:16], r, np.arange(32))
    assert out['model_vs_reference_a']['increment_power_relative'] == 0
    assert out['model_vs_reference_b'] == out['reference_a_vs_b']
    assert out['reference_b_vs_a']['increment_power_relative'] > 0


def test_reject_pseudoreplicates_and_mismatched_parents():
    r = paths()
    with pytest.raises(ValueError):
        paired_comparisons(r[:16], r, np.zeros(32))
    with pytest.raises(ValueError):
        paired_comparisons(r[:8], r, np.arange(32))
    g = r[:16].copy(); g[:, 0] = 1
    with pytest.raises(ValueError):
        paired_comparisons(g, r, np.arange(32))
