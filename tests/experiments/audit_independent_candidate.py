"""Audit validation noise separation and nearest actual training-window states."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from scripts.workflow.bauer_campaign import digest, save
from tests.experiments.coverage_transition_dataset import CoverageTransitionDataset


def reference_seeds(cfg, case, anchor, anchors):
    return cfg['noise_base'] + (case * anchors + anchor) * 10000 + np.arange(cfg['paths_per_anchor'])


def run():
    root = Path('tests/results/independent_candidate_reference')
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    cfg = manifest['config']
    assert digest('tests/conf/independent_candidate_validation.json') == manifest['config_sha256']
    bank = Path('tests/data/independent_candidate_reference/anchors.npz')
    assert digest(bank) == manifest['bank_sha256']
    with np.load(bank) as z:
        states = z['unit_initials'].copy()
    planned = np.concatenate([reference_seeds(cfg, c, a, len(manifest['anchors']))
                              for c, a in manifest['schedule']])
    assert len(planned) == manifest['total_paths'] == len(np.unique(planned))
    checked = []
    sources = [('fine_transition_pilot', 'case*_anchor*.npz', 48),
               ('coverage_intervention', 'anchor*.npz', 24),
               ('fresh_rollout_reference', 'case*_anchor*.npz', 36),
               ('conditional_restart', 'reference_case*_state*.npz', 180)]
    for directory, pattern, expected in sources:
        paths = sorted((Path('tests/data') / directory).glob(pattern))
        assert len(paths) == expected, (directory, len(paths))
        for path in paths:
            with np.load(path) as z:
                old = z['seeds'].astype(np.int64)
            assert np.intersect1d(planned, old).size == 0, str(path)
            checked.append(dict(path=str(path), count=len(old),
                seed_array_sha256=hashlib.sha256(old.tobytes()).hexdigest()))
    dataset = CoverageTransitionDataset()
    endpoints = []
    for index in range(len(dataset)):
        sample = dataset[index]
        endpoints.extend([sample['initial'], sample['target']])
    endpoints = np.asarray(endpoints)
    # Include the site reversals used in augmentation; query raw and normalized states.
    endpoints /= np.linalg.norm(endpoints, axis=-1, keepdims=True)
    tree = cKDTree(endpoints.reshape(len(endpoints), -1))
    distances = np.minimum(tree.query(states.reshape(len(states), -1))[0],
                           tree.query(states[:, ::-1].reshape(len(states), -1))[0])
    assert np.min(distances) > 1e-5, 'Validation initial equals a training endpoint within rounding tolerance'
    save(root / 'isolation_audit.json', dict(status='NO_OVERLAP_IN_AUDITED_SOURCES',
        manifest_sha256=digest(manifest_path), code_sha256=digest(__file__),
        planned_reference_seed_count=len(planned), checked_seed_files=checked,
        training_pairs=len(dataset), training_endpoint_count=len(endpoints),
        minimum_endpoint_distances=distances.tolist(),
        scope='Audit covers actual seed arrays in four named datasets and all 12288 mixed training pair endpoints, including site reversal. Does not claim audit of every historical file, PRNG stream independence from unequal integer seeds, or distributional novelty. Distances exclude duplicates, not certify sufficient coverage.'))
    print('Audited', len(checked), 'seed files;', len(endpoints), 'training endpoints; minimum distance', distances.min())


if __name__ == '__main__':
    run()
