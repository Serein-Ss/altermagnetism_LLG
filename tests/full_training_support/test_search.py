import numpy as np
import pytest
from tests.local_support_diagnosis.metrics import nearest, patches
from .search import ExactSearch, patch_batch, eligible_frames


def test_eligible_grid_excludes_target_only_endpoint():
    assert np.array_equal(eligible_frames(range(0, 991, 10), 10, 1001), np.arange(1000))
    assert np.array_equal(eligible_frames([0], 10, 11), np.arange(10))
    with pytest.raises(ValueError):
        eligible_frames([1], 10, 11)


def test_streaming_exact_matches_brute_and_provenance():
    rng = np.random.default_rng(52)
    a = rng.normal(size=(7, 16, 3)); a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b = rng.normal(size=(25, 3)); b /= np.linalg.norm(b, axis=-1, keepdims=True)
    for radius in (1, 2, 4):
        query, masks = patches(b, radius)
        bank, bm = patch_batch(a, radius)
        search = ExactSearch(query, masks)
        split = 30
        search.update(bank[:split], bm[:split], lambda i: i)
        search.update(bank[split:], bm[split:], lambda i: i+split)
        assert np.allclose(search.distances(), nearest(query, masks, bank, bm), atol=1e-12)
        for j, i in enumerate(search.locations):
            assert np.array_equal(bm[i], masks[j])
            assert nearest(query[j:j+1], masks[j:j+1], bank[i:i+1], bm[i:i+1])[0] == pytest.approx(search.distances()[j], abs=1e-12)


def test_missing_mask_rejected_and_append_cannot_increase_distance():
    state = np.tile([0., 0., 1.], (16, 1))
    q, m = patches(state, 2); search = ExactSearch(q, m)
    search.update(q[3:4], m[3:4], lambda i: i)
    with pytest.raises(ValueError, match='mask'):
        search.distances()
    search.update(q, m, lambda i: i)
    before = search.distances().copy()
    search.update(q, m, lambda i: i)
    assert np.all(search.distances() <= before+1e-12)
