import json
import numpy as np
from tests.experiments.fresh_rollout_reference import initial_bank
from tests.experiments.independent_reference import validate_inputs


def test_fresh_bank_reproducible_and_grids_valid():
    cfg = json.load(open('tests/conf/fresh_rollout_reference.json'))
    states, rows = initial_bank(cfg)
    np.testing.assert_array_equal(states, initial_bank(cfg)[0])
    assert states.shape == (6, 25, 3) and len(rows) == 6
    assert len({s.tobytes() for s in states}) == 6
    np.testing.assert_allclose(np.linalg.norm(states, axis=-1), 1, atol=1e-14)
    all_seeds = []
    for ci, case in enumerate(cfg['cases']):
        for ai in range(12):
            seeds = cfg['noise_base']+(ci*12+ai)*10000+np.arange(256)
            all_seeds.extend(seeds)
        _, _, steps, stride = validate_inputs(states, np.arange(6), **case,
            duration=cfg['duration'], save_dt=cfg['save_dt'], anisotropy=cfg['anisotropy'],
            damping=cfg['damping'], theta=.11)
        assert steps//stride == 80
    assert len(set(all_seeds)) == 9216
