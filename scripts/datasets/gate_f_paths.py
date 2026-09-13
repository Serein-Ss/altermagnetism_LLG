"""Lazy one-path HDF5 reader for the Gate-F schema; legacy/pilots are rejected.

File datasets: spins [F,A,Nx,Ny,3], initial [A,Nx,Ny,3], time [F],
energy [F], neel [F,3]. Required attributes: schema='gate_f_path_v1',
complete=True, initial_id, noise_id, source_family_id, parent_trajectory_id,
initial_type, split, theta, alpha. Paths/IDs/hashes are frozen in the manifest.
"""
import json
import hashlib
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from scripts.validation.gate_f_contract import audit_records, digest


class GateFPaths(Dataset):
    def __init__(self, contract, split):
        rows = json.loads(Path(contract['data_manifest']['path']).read_text())['paths']
        audit_records(rows)
        initial_catalog = json.loads(Path(contract['initial_manifest']['path']).read_text())['initials']
        initial_lookup = {r['initial_id']: r for r in initial_catalog}
        if len(initial_lookup) != len(initial_catalog):
            raise ValueError('duplicate initial catalog IDs')
        self.rows = []
        layout = {}
        for row in rows:
            path = Path(row['path'])
            if digest(path) != row['sha256']:
                raise ValueError('path hash mismatch: '+str(path))
            with h5py.File(path) as h:
                if h.attrs.get('schema') != 'gate_f_path_v1' or not h.attrs.get('complete', False):
                    raise ValueError('only complete Gate-F paths accepted')
                for key in ('initial_id', 'noise_id', 'source_family_id', 'parent_trajectory_id', 'initial_type', 'split', 'theta', 'alpha'):
                    if h.attrs[key] != row[key]:
                        raise ValueError('metadata mismatch: '+key)
                f, a, nx, ny, xyz = h['spins'].shape
                if f != contract['frames'] or a != 2 or nx != ny or nx not in contract['model']['sizes'] or xyz != 3:
                    raise ValueError('path geometry outside contract')
                time = h['time'][:]
                if not np.isfinite(time).all() or not np.isclose(time[0], 0) or not np.allclose(np.diff(time), contract['save_dt']):
                    raise ValueError('physical time grid outside contract')
                if row['theta'] not in contract['theta'] or row['alpha'] != contract['model']['alpha']:
                    raise ValueError('condition outside contract')
                if not np.array_equal(h['spins'][0], h['initial'][:]):
                    raise ValueError('initial frame mismatch')
                initial = h['initial'][:]
                sha = hashlib.sha256(initial.tobytes(order='C')).hexdigest()
                catalog = initial_lookup[row['initial_id']]
                if sha != catalog['spins_sha256'] or list(initial.shape) != catalog['shape'] or str(initial.dtype) != catalog['dtype']:
                    raise ValueError('initial catalog mismatch')
                if catalog['split'] != row['split'] or catalog['source_family_id'] != row['source_family_id']:
                    raise ValueError('initial provenance mismatch')
                if row['initial_type'] not in contract['initial_types']:
                    raise ValueError('unregistered initial type')
                key = (nx, row['theta'], row['initial_type'])
                layout.setdefault(key, {}).setdefault((row['initial_id'], row['split']), []).append(row['noise_id'])
                if h['energy'].shape != (f,) or h['neel'].shape != (f, 3):
                    raise ValueError('observable shape mismatch')
            if row['split'] == split:
                self.rows.append(row)
        expected = {(size, theta, kind) for size in contract['model']['sizes'] for theta in contract['theta'] for kind in contract['initial_types']}
        if set(layout) != expected:
            raise ValueError('incomplete Gate-F condition matrix')
        for group in layout.values():
            if len(group) != 4 or sum(split == 'train' for _, split in group) != 2:
                raise ValueError('requires two train and two development initials per group')
            if any(len(noises) != contract['noise_per_initial'] for noises in group.values()):
                raise ValueError('noise count outside initial Gate-F contract')
        if not self.rows:
            raise ValueError('empty split')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with h5py.File(row['path']) as h:
            spins = torch.from_numpy(h['spins'][:])
            if not torch.isfinite(spins).all() or (spins.norm(dim=-1)-1).abs().max() > (1e-10 if spins.dtype == torch.float64 else 1e-5):
                raise ValueError('nonfinite or nonunit trajectory')
            return dict(spins=spins, time=torch.from_numpy(h['time'][:]),
                        condition=spins.new_tensor([row['theta'], row['alpha']]), metadata=row)
