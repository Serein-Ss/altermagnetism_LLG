"""Freeze candidate provenance and prepare a disjoint validation initial bank."""
import json
from pathlib import Path

import numpy as np

from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.fresh_rollout_reference import initial_bank


def run():
    config = Path('tests/conf/independent_candidate_validation.json')
    cfg = json.loads(config.read_text())
    root = Path('tests/results/independent_candidate_reference')
    raw = Path('tests/data/independent_candidate_reference')
    if root.exists() or raw.exists():
        raise FileExistsError('Preserve existing validation; no silent regeneration')
    freeze = Path(cfg['candidate_freeze'])
    candidates = json.loads(freeze.read_text())
    assert candidates['status'] == 'FROZEN_DEVELOPMENT_CANDIDATE_NOT_CERTIFIED'
    models = []
    for model in candidates['checkpoints']:
        assert digest(model['checkpoint']) == model['sha256']
        report = Path(f"tests/results/reflection_augmentation_training/{model['seed']}/summary.json")
        assert digest(report) == model['report_sha256']
        models.append(dict(model, report=str(report)))
    assert {m['seed'] for m in models} == {11, 22, 33}
    states, families = initial_bank(cfg)
    assert states.shape == (12, 25, 3) and np.isfinite(states).all()
    np.testing.assert_allclose(np.linalg.norm(states, axis=-1), 1, atol=1e-14)
    old_banks = ['tests/data/fine_transition_pilot/anchors.npz',
                 'tests/data/fresh_rollout_reference/anchors.npz',
                 'tests/data/coverage_intervention/initials.npz']
    checked = {}
    for path in old_banks:
        with np.load(path) as z:
            arrays = [z[k] for k in z.files if z[k].ndim == 3 and z[k].shape[1:] == (25, 3)]
            assert arrays, f'No initial-state array in {path}'
            for old in arrays:
                assert all(not np.array_equal(a, b) for a in states for b in old)
        checked[path] = digest(path)
    assert len(np.unique(states.reshape(12, -1), axis=0)) == 12
    anchors = [dict(row, theta=theta) for theta in cfg['theta'] for row in families]
    schedule = [(c, a) for c in range(3) for a in range(len(anchors))]
    order = np.random.default_rng(cfg['schedule_seed']).permutation(len(schedule))
    root.mkdir(); raw.mkdir()
    bank = raw / 'anchors.npz'
    np.savez_compressed(bank, unit_initials=states)
    save(root / 'manifest.json', dict(config=cfg, config_sha256=digest(config),
        anchors=anchors, bank_sha256=digest(bank), frozen_models=models,
        candidate_freeze_sha256=digest(freeze), checked_old_banks=checked,
        schedule=[schedule[i] for i in order], total_paths=len(schedule)*cfg['paths_per_anchor'],
        code_sha256={p: digest(p) for p in [__file__, 'tests/experiments/fresh_rollout_reference.py']},
        status='INITIAL_BANK_PREPARED_NO_PATHS_GENERATED',
        limitations='Exact initial-state nonoverlap checked; not proof of distributional independence or complete training-window exclusion. Noise-seed audit and simulator/sampler preflight still required before job submission.'))
    print(f'Prepared {len(anchors)} conditions; {len(schedule)*cfg["paths_per_anchor"]} planned reference paths.')


if __name__ == '__main__':
    run()
