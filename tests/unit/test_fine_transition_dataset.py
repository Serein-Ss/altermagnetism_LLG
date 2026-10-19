import json
import numpy as np
import pytest
from tests.experiments.fine_transition_dataset import FineTransitionDataset


def test_real_pilot_family_split_and_exact_pairs():
    train = FineTransitionDataset('train')
    dev = FineTransitionDataset('development')
    assert set(train.parents).isdisjoint(dev.parents)
    assert train._cache is None and dev._cache is None
    assert len(train) == len(dev) == 12288
    seen = []
    for dataset in (train, dev):
        keys = set()
        for group, row in enumerate(dataset.rows):
            first = group*dataset.npaths*len(dataset.starts)
            sample = dataset[first]
            with np.load(row['path']) as d:
                np.testing.assert_array_equal(sample['initial'], d['spins'][0, 0])
                np.testing.assert_array_equal(sample['target'], d['spins'][0, 4])
                keys.update((sample['parent'], int(seed)) for seed in d['seeds'])
            sample['initial'][:] = 0
            assert np.linalg.norm(dataset[first]['initial'][0]) > .99
            last = dataset[first+dataset.npaths*len(dataset.starts)-1]
            assert last['start'] == 28
        seen.append(keys)
    assert seen[0].isdisjoint(seen[1])
    with pytest.raises(IndexError):
        train[len(train)]
    with pytest.raises(ValueError):
        FineTransitionDataset('test')
    with pytest.raises(ValueError):
        FineTransitionDataset('train', lag=.003)
    with pytest.raises(ValueError):
        FineTransitionDataset('train', lag=float('inf'))


def test_changed_data_rejected(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    original = 'tests/results/fine_transition_pilot/'
    from pathlib import Path
    manifest = Path(original+'manifest.json').read_text()
    (source/'manifest.json').write_text(manifest)
    report = json.loads(Path(original+'summary.json').read_text())
    for row in report['rows']:
        row['sha256'] = 'invalid'
    (source/'summary.json').write_text(json.dumps(report))
    dataset = FineTransitionDataset('train', source=source)
    with pytest.raises(ValueError, match='changed data'):
        dataset[0]
