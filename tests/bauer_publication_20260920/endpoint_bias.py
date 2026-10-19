"""Descriptive signed magnetization bias; first step shares the real initial state."""
import json
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent


def compare(generated, reference):
    a = np.asarray(generated, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.ndim != 4 or b.ndim != 4 or a.shape[1:] != b.shape[1:]:
        raise ValueError('Expected matching branch,time,site,component arrays')
    if a.shape[-1] != 3 or min(len(a), len(b)) < 2:
        raise ValueError('Need 3 components and multiple branches')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite trajectory')
    if not np.allclose(a[:, 0], b[0, 0]) or not np.allclose(b[:, 0], b[0, 0]):
        raise ValueError('All branches must share the same initial state')
    ma, mb = a[..., 2].mean(axis=2), b[..., 2].mean(axis=2)
    return dict(signed_bias=(ma.mean(axis=0)-mb.mean(axis=0)).tolist(),
                branch_sampling_se=np.sqrt(ma.var(axis=0, ddof=1)/len(a)
                                          + mb.var(axis=0, ddof=1)/len(b)).tolist())


def main():
    rows = []
    for seed in [11, 22, 33]:
        folder = HERE.parent/'bauer_transition_20260920'/'evaluation'/f'transition_{seed}'
        assert json.loads((folder/'complete.json').read_text())['model'] == f'transition_{seed}'
        for length in [16, 25, 32, 50, 100]:
            with h5py.File(folder/f'L{length}.h5', 'r') as g:
                assert g.attrs['complete']
                times = g['time_ps'][:]
                for parent in range(6):
                    path = HERE.parent/'bauer_two_stage_20260919'/'data'/f'L{length}'/f'test_future_{parent}.h5'
                    with h5py.File(path, 'r') as r:
                        assert r.attrs['complete']
                        assert np.allclose(times, r['time_ps'][:len(times)])
                        rows.append(dict(seed=seed, L=length, parent=parent,
                                         **compare(g[f'parent_{parent}'][:], r['spins'][:, :len(times)])))
    result = dict(time_ps=times.tolist(), rows=rows, scientific_certification=False,
                  interpretation='Exploratory; branch SE conditional on each fixed parent/checkpoint. '
                  'No confidence band, multiplicity correction, or training-seed uncertainty. '
                  'Only the first transition uses a shared real state; later bias is rollout bias, not local drift.')
    (HERE/'results'/'endpoint_bias.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for seed in [11, 22, 33]:
        a = [r for r in rows if r['seed'] == seed and r['L'] == 16]
        print(seed, 'L16 mean bias at 0.1,1,10 ps', np.mean([r['signed_bias'] for r in a], axis=0)[[1, 10, 100]])


if __name__ == '__main__':
    main()
