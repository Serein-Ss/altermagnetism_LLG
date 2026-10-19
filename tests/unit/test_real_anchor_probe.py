import numpy as np
import torch
import pytest
from tests.experiments.real_anchor_probe import generate


class ZeroField:
    def __call__(self, x, *args, **kwargs):
        return torch.zeros_like(x)


def test_distinct_anchors_replay_diversity_and_no_future_argument():
    anchors = np.random.default_rng(21).normal(size=(4, 5, 3))
    anchors /= np.linalg.norm(anchors, axis=-1, keepdims=True)
    cfg = dict(batch_size=2, frames=11, steps=2, latent_base=870000000)
    a, seeds = generate(ZeroField(), anchors, .11, cfg, 0, 0, 'cpu')
    b, _ = generate(ZeroField(), anchors, .13, cfg, 0, 0, 'cpu')
    assert a.shape == (4, 11, 5, 3) and seeds == [870000000, 870000001]
    np.testing.assert_array_equal(a[:, 0], anchors.astype('f4'))
    np.testing.assert_array_equal(a, b)
    c, _ = generate(ZeroField(), anchors, .11, cfg, 0, 1, 'cpu')
    assert not np.array_equal(a[:, 1:], c[:, 1:])
    with pytest.raises(ValueError):
        generate(ZeroField(), anchors*2, .11, cfg, 0, 0, 'cpu')
