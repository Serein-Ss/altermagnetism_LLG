import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.experiments.audit_independent_candidate import reference_seeds
from tests.experiments.independent_candidate_sampling import rollout_config
from tests.experiments.independent_reference import generate
from tests.experiments.director_midpoint_sampling import rollout


def test_registered_seed_and_time_mapping():
    cfg = json.loads(Path('tests/conf/independent_candidate_validation.json').read_text())
    seeds = np.concatenate([reference_seeds(cfg, c, a, 24) for c in range(3) for a in range(24)])
    assert len(seeds) == len(np.unique(seeds)) == 36864
    training = {'lag': .01, 'axis': [0, 0, 1]}
    sample = rollout_config(training, cfg, cfg['sampling'])
    assert sample['rollout_steps'] == 80 and sample['generated_paths'] == 512
    assert sample['generation_batch'] == 64
    assert training == {'lag': .01, 'axis': [0, 0, 1]}


@pytest.mark.parametrize('method,dt', [('mt', .0025), ('mt', .00125), ('heun', .00125)])
def test_synthetic_reference_shapes_and_unit_constraint(method, dt):
    states = np.zeros((2, 25, 3)); states[..., 2] = 1
    p, diagnostics = generate(states, np.array([12341, 12342]), method=method, dt=dt,
        duration=.02, save_dt=.01, anisotropy=.1, damping=.1, theta=.11)
    assert p.shape == (2, 3, 25, 3)
    np.testing.assert_array_equal(p[:, 0], states)
    np.testing.assert_allclose(np.linalg.norm(p, axis=-1), 1, atol=1e-12)
    assert diagnostics['device'] == 'cpu'


def test_synthetic_sampler_replay_and_shape():
    class ZeroField:
        def __call__(self, x, *args):
            return torch.zeros_like(x)
    initial = np.zeros((25, 3)); initial[:, 2] = 1
    cfg = dict(generated_paths=3, generation_batch=2, rollout_steps=2, lag=.01, axis=[0, 0, 1])
    p, state = rollout(ZeroField(), initial, .11, cfg, .016, 32, 12343, 'cpu')
    replay, other = rollout(ZeroField(), initial, .11, cfg, .016, 32, 12343, 'cpu')
    assert p.shape == (3, 3, 25, 3) and np.isfinite(p).all()
    np.testing.assert_array_equal(p, replay)
    np.testing.assert_array_equal(state, other)
    np.testing.assert_allclose(np.linalg.norm(p, axis=-1), 1, atol=1e-6)


def test_production_entrypoints_require_scheduler(monkeypatch):
    from tests.experiments.independent_candidate_reference import run as reference_run
    from tests.experiments.independent_candidate_sampling import run as sampling_run
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(RuntimeError, match='Slurm'):
        reference_run()
    with pytest.raises(RuntimeError, match='Slurm'):
        sampling_run(11, 'cpu')
