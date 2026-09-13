"""Lazy full-space reader. Keeps the actual two/four-site basis, no crops.

Diagnostic shards require opt-in and cannot enter a training split. Production
support is deliberately fail-closed until the independent physics gate exists.
"""
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ZeroFieldPaths(Dataset):
    def __init__(self,paths,*,allow_diagnostic=False):
        self.rows=[]
        for path in map(Path,paths):
            with h5py.File(path) as h:
                if not h.attrs.get('complete',False): raise ValueError('incomplete shard')
                if h.attrs.get('schema')!='gomonay_zero_field_v1_diagnostic': raise ValueError('unsupported schema')
                if not allow_diagnostic: raise ValueError('diagnostic calibration cannot enter model training')
                g=json.loads(h.attrs['geometry_json'])
                if h['spins'].shape[-2]!=g['basis']: raise ValueError('basis mismatch')
                if not np.all(np.diff(h['time'][:])>0): raise ValueError('time must be physical and increasing')
                self.rows.extend((path,i) for i in range(h['spins'].shape[1]))

    def __len__(self): return len(self.rows)

    def __getitem__(self,index):
        path,i=self.rows[index]
        with h5py.File(path) as h:
            return dict(spins=torch.from_numpy(h['spins'][:,i]),
                initial_state=torch.from_numpy(h['initial_state'][i]),
                time=torch.from_numpy(h['time'][:]),mask=torch.from_numpy(h['mask'][:,i]),
                geometry=json.loads(h.attrs['geometry_json']),
                sublattice_sign=torch.from_numpy(h['sublattice_sign'][:]),
                coordinates=torch.from_numpy(h['coordinates'][:]),
                conditions=json.loads(h.attrs['protocol_json']),
                parent_chain_id=h['parent_chain_id'][i].decode(),
                initial_state_id=h['initial_state_id'][i].decode(),
                noise_seed=int(h['noise_seed'][i]),sample_weight=float(h['sample_weight'][i]),
                split=h.attrs['split'],events_status=h.attrs['events_status'])


def audit_group_splits(records):
    """Both preparation chains and repeated initial states must stay together."""
    seen={}
    for row in records:
        for field in ('parent_chain_id','initial_state_id'):
            key=(field,row[field])
            if key in seen and seen[key]!=row['split']: raise ValueError(f'split leakage: {key}')
            seen[key]=row['split']
    return True
