import json
import numpy as np
import pytest
import torch
from tests.experiments.fresh_rollout_sampling import rollout_config, rollout


def test_duration_config_is_explicit_and_preserves_training():
    training = json.load(open('tests/conf/local_transition_training.json'))
    reference = json.load(open('tests/conf/fresh_rollout_reference.json'))
    sampling = json.load(open('tests/conf/fresh_rollout_sampling.json'))
    cfg = rollout_config(training, reference, sampling)
    assert cfg['rollout_steps'] == 80 and training['rollout_steps'] == 8
    with pytest.raises(ValueError):
        rollout_config(training, dict(reference, duration=.805), sampling)
    with pytest.raises(ValueError):
        rollout_config(training, dict(reference, save_dt=.02), sampling)


def test_long_rollout_shape_norm_and_paired_rng():
    class ZeroField:
        def __call__(self, x, *args):
            return torch.zeros_like(x)
    cfg = dict(generated_paths=4, generation_batch=2, rollout_steps=80, lag=.01, axis=[0.,0.,1.])
    original = np.tile([0.,0.,1.], (3,1))
    a, ra = rollout(ZeroField(), original, .11, cfg, .02, 2, 7, 'cpu')
    b, rb = rollout(ZeroField(), original, .11, cfg, .02, 4, 7, 'cpu')
    assert a.shape == b.shape == (4,81,3,3)
    np.testing.assert_array_equal(ra, rb)
    np.testing.assert_allclose(np.linalg.norm(a, axis=-1), 1, atol=2e-7)
    assert not np.array_equal(a[0], a[1])
