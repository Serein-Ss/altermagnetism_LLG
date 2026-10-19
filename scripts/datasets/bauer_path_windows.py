"""Lazy, physical-time windows for the frozen Bauer branching pilot.

Only base-arm futures are model samples. This reader validates the storage
contract, not numerical accuracy or the physical distribution.
"""
import json
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset


class PathWindows(Dataset):
    def __init__(self, roots, split, duration_ps, stride_frames=1):
        if split not in ('train', 'dev', 'test'):
            raise ValueError('split must be train, dev or test')
        if duration_ps <= 0 or not isinstance(stride_frames, int) or stride_frames < 1:
            raise ValueError('positive duration and integer stride required')
        self.records = []
        self.ends = []
        parents = {}
        total = 0
        seen = set()
        for root in roots:
            root = Path(root).resolve()
            if not (root / 'complete.json').exists():
                raise ValueError(f'incomplete dataset: {root}')
            for row in json.loads((root / 'manifest.json').read_text()):
                parent = row['parent_id']
                if parent in parents and parents[parent] != row['split']:
                    raise ValueError(f'parent split leakage: {parent}')
                parents[parent] = row['split']
                if row['split'] != split or row['arm'] != 'base':
                    continue
                path = (root / row['file']).resolve()
                if not path.is_relative_to(root) or path in seen:
                    raise ValueError('duplicate or out-of-root sample file')
                seen.add(path)
                with h5py.File(path, 'r') as h:
                    if not h.attrs.get('complete', False):
                        raise ValueError(f'incomplete sample: {path}')
                    times = h['time_ps'][:]
                    shape = h['spins'].shape
                    meta = json.loads(h.attrs['parent_json'])
                    if meta['parent_id'] != parent or meta['split'] != split:
                        raise ValueError('manifest/HDF5 parent mismatch')
                    if h.attrs['arm'] != 'base' or len(shape) != 4 or shape[-1] != 3:
                        raise ValueError('expected base [branch, time, site, xyz]')
                    if shape[:2] != (row['branches'], row['frames']) or len(times) != shape[1]:
                        raise ValueError('manifest/HDF5 shape mismatch')
                    dt = times[1] - times[0]
                    if dt <= 0 or not np.allclose(np.diff(times), dt, rtol=1e-7, atol=1e-10):
                        raise ValueError('expected uniform physical time grid')
                    steps = round(duration_ps / dt)
                    if steps < 1 or not np.isclose(steps * dt, duration_ps, rtol=1e-7, atol=1e-10):
                        raise ValueError('duration is not on saved physical time grid')
                    if steps >= len(times):
                        raise ValueError('requested window exceeds stored future')
                starts = range(0, len(times) - steps, stride_frames)
                total += shape[0] * len(starts)
                self.records.append((path, row, starts, steps))
                self.ends.append(total)
        if not self.records:
            raise ValueError('no matching base-arm samples')

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        group = int(np.searchsorted(self.ends, index, side='right'))
        offset = index - (self.ends[group - 1] if group else 0)
        path, row, starts, steps = self.records[group]
        branch, window = divmod(offset, len(starts))
        start = starts[window]
        with h5py.File(path, 'r') as h:
            spins = h['spins'][branch, start:start + steps + 1]
            times = h['time_ps'][start:start + steps + 1]
            seed = int(h['seeds'][branch])
        if not np.isfinite(spins).all() or not np.allclose(np.linalg.norm(spins, axis=-1), 1, atol=1e-6, rtol=0):
            raise ValueError(f'invalid spin sample: {path}, branch {branch}')
        return dict(spins=spins, initial=spins[0].copy(), time_ps=times-times[0],
                    origin_ps=float(times[0]), parent_id=row['parent_id'],
                    family=row['family'], split=row['split'], branch=branch, seed=seed)
