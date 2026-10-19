"""Lazy same-initial-state branch pairs, restricted to the training split."""
import json
import h5py
import numpy as np
from tests.mdgen_path_refactor_20260918.data import PathWindows


def read_groups(path, parent_id, rng, branches):
    with h5py.File(path, 'r') as h:
        meta = json.loads(h.attrs['parent_json'])
        if not h.attrs['complete'] or meta['split'] != 'train' or meta['parent_id'] != parent_id or h.attrs['arm'] != 'base':
            raise ValueError('Only complete matching training parents are allowed')
        seeds = h['seeds'][:]
        if len(np.unique(seeds)) != len(seeds):
            raise ValueError('Duplicate noise seeds')
        if branches < 1 or 2*branches > len(seeds):
            raise ValueError('Need two disjoint reference branch groups')
        times = h['time_ps'][:2]
        if not np.allclose(times-times[0], [0., .1], atol=1e-8, rtol=0):
            raise ValueError('Expected a 0.1 ps transition')
        # Only the first transition has a common state across independent branches.
        x = h['spins'][:, :2].astype('f4')
    if not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x, axis=-1), 1, atol=1e-6, rtol=0):
        raise ValueError('Invalid spins')
    if not np.allclose(x[:, 0], x[0, 0], atol=2e-7, rtol=0):
        raise ValueError('Branches have different initial states')
    idx = rng.permutation(len(x))[:2*branches]
    return dict(initial=x[0, 0], reference_a=x[idx[:branches], 1],
                reference_b=x[idx[branches:], 1], seeds_a=seeds[idx[:branches]],
                seeds_b=seeds[idx[branches:]], parent_id=parent_id)


class TrainingBranchPairs:
    def __init__(self, roots):
        validated = PathWindows(roots, 'train', .1)
        self.records = [(path, row) for path, row, _, _ in validated.records]

    def sample(self, index, rng, branches=4):
        path, row = self.records[index]
        return read_groups(path, row['parent_id'], rng, branches)

    def __len__(self):
        return len(self.records)
