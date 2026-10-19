import json
import h5py
import numpy as np
import pytest
from tests.bauer_moment_training_20260920.data import read_groups


def test_split_disjointness_and_shared_initial(tmp_path):
    path = tmp_path/'branches.h5'
    with h5py.File(path, 'w') as h:
        h.attrs['complete'] = True; h.attrs['arm'] = 'base'
        h.attrs['parent_json'] = json.dumps(dict(parent_id='p0', split='train'))
        h['seeds'] = np.arange(8); h['time_ps'] = [0., .1]
        x = np.zeros((8, 2, 3, 3), dtype='f4'); x[..., 2] = 1
        h['spins'] = x
    r = read_groups(path, 'p0', np.random.default_rng(3), 4)
    assert not set(r['seeds_a']) & set(r['seeds_b'])
    with h5py.File(path, 'r+') as h:
        h.attrs['parent_json'] = json.dumps(dict(parent_id='p0', split='test'))
    with pytest.raises(ValueError, match='training parents'):
        read_groups(path, 'p0', np.random.default_rng(3), 4)
    with h5py.File(path, 'r+') as h:
        h.attrs['parent_json'] = json.dumps(dict(parent_id='p0', split='train'))
        h['spins'][1, 0, 0] = [1., 0., 0.]
    with pytest.raises(ValueError, match='different initial'):
        read_groups(path, 'p0', np.random.default_rng(3), 4)
