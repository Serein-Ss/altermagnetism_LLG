"""Lazy pilot transition pairs; family-separated development, not a test set."""
import json
from pathlib import Path
import numpy as np
from scripts.workflow.bauer_campaign import digest


class FineTransitionDataset:
    def __init__(self, split, source='tests/results/fine_transition_pilot', lag=.01):
        if split not in ('train', 'development'):
            raise ValueError('only prototype train/development splits exist')
        source = Path(source)
        report = json.loads((source/'summary.json').read_text())
        manifest = json.loads((source/'manifest.json').read_text())
        if report['status'] != 'COMPLETE_FINE_TRANSITION_PILOT_NOT_CERTIFIED':
            raise ValueError('incomplete source')
        if digest(source/'manifest.json') != report['manifest_sha256']:
            raise ValueError('changed manifest')
        cfg = manifest['config']; dt = cfg['cases'][2]['dt']
        if not np.isfinite(lag) or lag <= 0:
            raise ValueError('lag must be finite and positive')
        step = round(lag/dt)
        if step < 1 or not np.isclose(step*dt, lag, atol=1e-12, rtol=0):
            raise ValueError('lag must align with saved grid')
        frames = round(cfg['duration']/dt)+1
        self.starts = np.arange(0, frames-step, step)
        if not len(self.starts):
            raise ValueError('lag exceeds observed duration')
        self.step, self.lag = step, lag
        self.npaths, self.frames = cfg['paths_per_anchor'], frames
        family = 0 if split == 'train' else 1
        self.rows = []
        for row in sorted(report['rows'], key=lambda r: r['anchor']):
            state = manifest['anchors'][row['anchor']]
            if row['case'] == 2 and state['family'] == family:
                self.rows.append(dict(row, state=state))
        if len(self.rows) != 6 or len({r['anchor'] for r in self.rows}) != 6:
            raise ValueError('incomplete prototype split')
        self.parents = sorted({r['state']['parent'] for r in self.rows})
        self._cache_index = None
        self._cache = None

    def __len__(self):
        return len(self.rows)*self.npaths*len(self.starts)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        group, offset = divmod(index, self.npaths*len(self.starts))
        path_index, window = divmod(offset, len(self.starts))
        row = self.rows[group]
        if self._cache_index != group:
            if digest(row['path']) != row['sha256']:
                raise ValueError('changed data file')
            with np.load(row['path']) as data:
                spins, times, seeds = data['spins'], data['reduced_time'], data['seeds']
            if spins.shape[:2] != (self.npaths, self.frames) or spins.ndim != 4 or spins.shape[-1] != 3:
                raise ValueError('invalid trajectory shape')
            if not np.isfinite(spins).all() or not np.allclose(np.linalg.norm(spins, axis=-1), 1, atol=1e-10, rtol=0):
                raise ValueError('invalid spins')
            np.testing.assert_allclose(times, np.arange(self.frames)*(self.lag/self.step), atol=1e-12, rtol=0)
            if seeds.shape != (self.npaths,) or len(np.unique(seeds)) != self.npaths:
                raise ValueError('invalid noise identities')
            self._cache = spins, seeds
            self._cache_index = group
        spins, seeds = self._cache
        start = int(self.starts[window])
        return dict(initial=spins[path_index, start].copy(),
                    target=spins[path_index, start+self.step].copy(),
                    theta=float(row['state']['theta']), lag=float(self.lag),
                    parent=int(row['state']['parent']), anchor=int(row['anchor']),
                    noise_seed=int(seeds[path_index]), start=start)
